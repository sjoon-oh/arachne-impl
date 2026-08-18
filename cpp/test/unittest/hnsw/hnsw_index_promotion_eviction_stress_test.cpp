// "Hard" stress tests for the HNSW port under promotion/eviction churn --
// distinct from hnsw_index_dist_test.cpp/hnsw_index_anchor_entry_dist_test.cpp
// (which use a generous GPU budget so nothing ever gets evicted mid-test).
// Here the GPU budget is deliberately too small for the whole dataset, and
// many threads hammer Controller::search() concurrently, so RegionManager's
// Coordinator is under constant promotion/eviction pressure the whole run.
//
// Also covers two specific risks this port's existing tests never exercised
// (identified by reading Controller::dispatch(const ModifyRequest&) and
// RegionManager::clearResidency()'s call sites -- see this file's own
// comments below for what was actually found):
//   - Whether heavy concurrent promotion/eviction churn alone can corrupt
//     results or crash the process (thread-safety of the coarse mutex_ +
//     Lease/pin machinery under real contention, not just sequential calls).
//   - Whether a host-side Insert (modifyHost(), the only Insert path this
//     port has -- modifyDevice() is unimplemented by design) into a Region
//     that's *already* GPU-resident leaves a stale device copy that
//     traverseDevice() can silently keep serving, since
//     Controller::dispatch(const ModifyRequest&) has no on_complete hook at
//     all (unlike the TraverseRequest overload) -- nothing in Core ever
//     calls RegionManager::clearResidency() in reaction to a completed
//     Insert/Delete.

#include "hnsw_index_anchor_entry.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "logging.hpp"
#include "stress_test_support.hpp"

namespace {

using arachne::ASRoutingCacheHnsw;
using arachne::Controller;
using arachne::ControllerStats;
using arachne::DistanceMetric;
using arachne::ModifyOp;
using arachne::ModifyRequest;
using arachne::Query;
using arachne::Record;
using arachne::RegionId;
using arachne::SchedulingConfig;
using arachne::SearchResult;
using arachne::TraverseRequest;
using arachne::TraverseResult;
using arachne::VectorBatchView;
using arachne::VectorDType;
using arachne::VectorId;
using arachne::VectorView;
using arachne::index::hnsw::HnswIndexAnchorEntry;
using arachne::stress::testsupport::GenerateVectors;

constexpr std::uint32_t kDim = 16;
constexpr std::size_t kVectorsPerRegion = 32;
constexpr std::size_t kM = 16;
constexpr std::size_t kEfConstruction = 100;

std::size_t Overlap(const TraverseResult& a, const TraverseResult& b) {
	std::unordered_set<VectorId> ids;
	for (const auto& n : a.result.neighbors) ids.insert(n.id);
	std::size_t overlap = 0;
	for (const auto& n : b.result.neighbors) {
		if (ids.count(n.id)) ++overlap;
	}
	return overlap;
}

// Runs `fn(thread_index)` on `num_threads` std::threads, collecting any
// exception thrown on any thread and re-throwing the first one on the
// calling (test) thread -- so a worker-thread crash/exception fails the
// test loudly instead of silently terminating (default std::thread
// behavior) or being swallowed.
void RunConcurrently(std::size_t num_threads, const std::function<void(std::size_t)>& fn) {
	std::vector<std::thread> threads;
	std::mutex exceptions_mutex;
	std::vector<std::exception_ptr> exceptions;
	for (std::size_t t = 0; t < num_threads; ++t) {
		threads.emplace_back([&, t] {
			try {
				fn(t);
			} catch (...) {
				std::lock_guard<std::mutex> lock(exceptions_mutex);
				exceptions.push_back(std::current_exception());
			}
		});
	}
	for (auto& thread : threads) thread.join();
	if (!exceptions.empty()) std::rethrow_exception(exceptions.front());
}

class HnswIndexPromotionEvictionStressTest : public testing::Test {
 protected:
	static constexpr std::size_t kNumVectors = 3000;
	static constexpr std::size_t kCapacity = 3500;  // headroom for inserts in later tests

	void SetUp() override {
		std::mt19937 rng(101);
		vectors_ = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

		index_ = std::make_unique<HnswIndexAnchorEntry>(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity,
																										 kVectorsPerRegion, kM, kEfConstruction);

		std::vector<std::byte> bytes(vectors_.front().size() * vectors_.size());
		std::vector<VectorId> ids(vectors_.size());
		for (std::size_t i = 0; i < vectors_.size(); ++i) {
			std::memcpy(bytes.data() + i * vectors_.front().size(), vectors_[i].data(), vectors_.front().size());
			ids[i] = static_cast<VectorId>(i) + 1;
		}
		index_->build(VectorBatchView{bytes.data(), kDim, VectorDType::Float32, ids.size(), ids.data()});

		total_host_bytes_ = 0;
		for (RegionId region : index_->allRegions()) total_host_bytes_ += index_->resolveRegion(region)->hostView().bytes;
	}

	std::vector<std::vector<std::byte>> vectors_;
	std::unique_ptr<HnswIndexAnchorEntry> index_;
	std::size_t total_host_bytes_ = 0;
};

// The "hard test" itself: a deliberately too-small GPU budget (1/4 of the
// whole dataset) plus 8 threads firing 3200 total searches at random
// localities. Every search is a promotion candidate; with only 25% of the
// dataset able to fit on GPU at once, the replacement policy is forced to
// evict continuously to make room for newer localities -- this is the
// "excessive promotion/eviction" scenario, not just a one-time settle.
TEST_F(HnswIndexPromotionEvictionStressTest, SurvivesHeavyConcurrentChurnAndStaysAccurate) {
	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/256, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/100, DistanceMetric::L2, VectorDType::Float32);
	SchedulingConfig scheduling_config;
	scheduling_config.max_execution_threads = 8;
	Controller controller(*index_, routing_cache, scheduling_config, nullptr, total_host_bytes_ / 4);
	index_->registerAllRegions(controller);
	index_->attachController(controller);

	constexpr std::size_t kThreads = 8;
	constexpr std::size_t kSearchesPerThread = 400;
	std::atomic<std::size_t> total_searches{0};

	RunConcurrently(kThreads, [&](std::size_t thread_index) {
		std::mt19937 rng(2000 + static_cast<unsigned>(thread_index));
		std::uniform_int_distribution<std::size_t> pick(0, vectors_.size() - 1);
		for (std::size_t i = 0; i < kSearchesPerThread; ++i) {
			std::size_t idx = pick(rng);
			Query query{VectorView{vectors_[idx].data(), kDim, VectorDType::Float32}, /*top_k=*/5};
			SearchResult result = controller.search(query);
			ASSERT_FALSE(result.neighbors.empty()) << "empty result for query index " << idx;
			total_searches.fetch_add(1, std::memory_order_relaxed);
		}
	});
	controller.waitIdle();

	ASSERT_EQ(total_searches.load(), kThreads * kSearchesPerThread);

	ControllerStats stats = controller.stats();
	ARACHNE_LOG_INFO(
			"SurvivesHeavyConcurrentChurnAndStaysAccurate: promoted={} evicted={} written_back={} anchor_evictions={} "
			"compactions={} gpu_bytes_allocated={}",
			stats.regions_promoted_total, stats.regions_evicted_total, stats.regions_written_back_total,
			stats.anchor_evictions_total, stats.compactions_total, stats.gpu_bytes_allocated);
	// With a 4x-oversubscribed budget and 3200 searches spread across the
	// whole dataset, both promotion AND eviction must have actually happened
	// -- if either is zero, the test isn't exercising what it claims to.
	EXPECT_GT(stats.regions_promoted_total, 0u);
	EXPECT_GT(stats.regions_evicted_total, 0u) << "budget was 1/4 of the dataset -- eviction should have been forced";

	// Accuracy spot-check post-churn: self-recall (querying with a point's
	// own vector) via Controller::search() should still find that point
	// itself with reasonably high probability, whether served GpuOnly or
	// Hybrid -- a corruption from lease/pin races or a use-after-free of
	// evicted device memory would show up here as wrong/garbage neighbors.
	std::size_t sample_count = 0;
	std::size_t self_hits = 0;
	for (std::size_t i = 0; i < vectors_.size(); i += 37) {
		Query query{VectorView{vectors_[i].data(), kDim, VectorDType::Float32}, /*top_k=*/1};
		SearchResult result = controller.search(query);
		ASSERT_FALSE(result.neighbors.empty());
		++sample_count;
		if (result.neighbors.front().id == static_cast<VectorId>(i) + 1) ++self_hits;
	}
	double self_recall = static_cast<double>(self_hits) / static_cast<double>(sample_count);
	ARACHNE_LOG_INFO("SurvivesHeavyConcurrentChurnAndStaysAccurate: post-churn self-recall {}/{} ({:.2f})", self_hits,
										sample_count, self_recall);
	EXPECT_GE(self_recall, 0.5) << "self-recall dropped to " << self_recall
															 << " after churn -- possible corruption, not just approximation";
}

// Same churn conditions, but with a subset of threads concurrently calling
// Controller::insert() for brand-new vectors while others keep searching --
// exercises the coarse mutex_ under real concurrent Insert+Search+
// promotion/eviction, not just concurrent Search alone above.
TEST_F(HnswIndexPromotionEvictionStressTest, ConcurrentInsertDuringChurnDoesNotCorruptOrCrash) {
	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/256, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/100, DistanceMetric::L2, VectorDType::Float32);
	SchedulingConfig scheduling_config;
	scheduling_config.max_execution_threads = 8;
	Controller controller(*index_, routing_cache, scheduling_config, nullptr, total_host_bytes_ / 4);
	index_->registerAllRegions(controller);
	index_->attachController(controller);

	std::mt19937 seed_rng(202);
	std::vector<std::vector<std::byte>> extra = GenerateVectors(VectorDType::Float32, kDim, 400, seed_rng);

	constexpr std::size_t kSearchThreads = 6;
	constexpr std::size_t kInsertThreads = 2;
	constexpr std::size_t kSearchesPerThread = 300;
	std::atomic<std::size_t> next_extra_index{0};
	std::atomic<std::size_t> inserts_ok{0};

	RunConcurrently(kSearchThreads + kInsertThreads, [&](std::size_t thread_index) {
		if (thread_index < kSearchThreads) {
			std::mt19937 rng(3000 + static_cast<unsigned>(thread_index));
			std::uniform_int_distribution<std::size_t> pick(0, vectors_.size() - 1);
			for (std::size_t i = 0; i < kSearchesPerThread; ++i) {
				std::size_t idx = pick(rng);
				Query query{VectorView{vectors_[idx].data(), kDim, VectorDType::Float32}, /*top_k=*/5};
				SearchResult result = controller.search(query);
				ASSERT_FALSE(result.neighbors.empty());
			}
		} else {
			for (;;) {
				std::size_t idx = next_extra_index.fetch_add(1, std::memory_order_relaxed);
				if (idx >= extra.size()) break;
				Record record;
				record.id = static_cast<VectorId>(kNumVectors) + static_cast<VectorId>(idx) + 1;
				record.vector = VectorView{extra[idx].data(), kDim, VectorDType::Float32};
				auto insert_result = controller.insert(record);
				if (insert_result.ok) inserts_ok.fetch_add(1, std::memory_order_relaxed);
			}
		}
	});
	controller.waitIdle();

	ARACHNE_LOG_INFO("ConcurrentInsertDuringChurnDoesNotCorruptOrCrash: inserts_ok={}/{}", inserts_ok.load(),
										extra.size());
	EXPECT_GT(inserts_ok.load(), 0u);
	EXPECT_EQ(index_->liveCount(), kNumVectors + inserts_ok.load());

	// Newly inserted vectors should be self-findable via a fresh host search
	// (ground truth -- this doesn't go through the possibly-stale device
	// path, just confirms the inserts themselves landed correctly under
	// concurrent load).
	std::size_t checked = 0;
	std::size_t found_self = 0;
	for (std::size_t i = 0; i < extra.size(); i += 20) {
		TraverseRequest request;
		request.query.vector = VectorView{extra[i].data(), kDim, VectorDType::Float32};
		request.query.top_k = 1;
		TraverseResult result = index_->traverseHost({request}).front();
		if (result.result.neighbors.empty()) continue;
		++checked;
		if (result.result.neighbors.front().id == static_cast<VectorId>(kNumVectors) + static_cast<VectorId>(i) + 1) {
			++found_self;
		}
	}
	ARACHNE_LOG_INFO("ConcurrentInsertDuringChurnDoesNotCorruptOrCrash: inserted-vector self-recall {}/{}", found_self,
										checked);
	EXPECT_GE(static_cast<double>(found_self) / static_cast<double>(checked), 0.5);
}

// Targeted, deterministic (not relying on concurrent timing) check for the
// specific risk flagged in this file's overview: does a host-side Insert
// into an already-GPU-resident Region leave traverseDevice() serving stale
// data for that Region afterward? Generous GPU budget (nothing gets evicted
// mid-test) isolates this from the churn/eviction scenarios above.
TEST(HnswIndexInsertAfterPromotionTest, DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot) {
	constexpr std::size_t kBaseCount = 300;
	constexpr std::size_t kCapacity = 800;

	std::mt19937 rng(303);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kBaseCount, rng);

	HnswIndexAnchorEntry index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
														 kEfConstruction);
	std::vector<std::byte> bytes(vectors.front().size() * vectors.size());
	std::vector<VectorId> ids(vectors.size());
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		std::memcpy(bytes.data() + i * vectors.front().size(), vectors[i].data(), vectors.front().size());
		ids[i] = static_cast<VectorId>(i) + 1;
	}
	index.build(VectorBatchView{bytes.data(), kDim, VectorDType::Float32, ids.size(), ids.data()});

	std::size_t total_host_bytes = 0;
	for (RegionId region : index.allRegions()) total_host_bytes += index.resolveRegion(region)->hostView().bytes;

	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/64, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/100, DistanceMetric::L2, VectorDType::Float32);
	// Generous budget -- every Region should fit; nothing should ever be
	// evicted here, so any staleness observed below can only be explained by
	// "insert wrote to a Region without invalidating its device copy", not by
	// ordinary eviction/re-promotion cycling.
	Controller controller(index, routing_cache, SchedulingConfig{}, nullptr, total_host_bytes * 4);
	index.registerAllRegions(controller);
	index.attachController(controller);

	// Promote as much of the dataset as possible by querying all of it.
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, /*top_k=*/3};
		controller.search(query);
	}
	controller.waitIdle();

	std::size_t resident_before = 0;
	for (RegionId region : index.allRegions()) {
		if (controller.acquireRegion(region).on_device) ++resident_before;
	}
	ASSERT_GT(resident_before, 0u) << "nothing got promoted -- test setup is broken";
	ARACHNE_LOG_INFO("DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot: {} region(s) resident before insert",
										resident_before);

	// Pick a query vector already in the dataset, and get *fresh* host ground
	// truth for it before touching anything else.
	std::size_t query_index = 5;
	TraverseRequest probe;
	probe.query.vector = VectorView{vectors[query_index].data(), kDim, VectorDType::Float32};
	probe.query.top_k = 5;
	TraverseResult before_insert = index.traverseHost({probe}).front();
	ASSERT_FALSE(before_insert.result.neighbors.empty());

	// Insert several near-duplicates of that same query vector (tiny random
	// perturbation) -- close enough that hnswlib's own algorithm should
	// place them as top-ranked neighbors of the query, and very likely as new
	// link-list neighbors of whatever pre-existing nodes were near the query
	// before (mutuallyConnectNewElement() rewiring existing nodes' level-0
	// link lists in their *own*, possibly-still-"resident" Regions).
	std::uniform_real_distribution<float> jitter(-0.01f, 0.01f);
	std::vector<VectorId> inserted_ids;
	for (int k = 0; k < 5; ++k) {
		std::vector<float> near(kDim);
		const auto* base = reinterpret_cast<const float*>(vectors[query_index].data());
		for (std::uint32_t d = 0; d < kDim; ++d) near[d] = base[d] + jitter(rng);
		VectorId new_id = static_cast<VectorId>(kBaseCount) + static_cast<VectorId>(k) + 1;
		Record record{new_id, VectorView{near.data(), kDim, VectorDType::Float32}};
		auto result = controller.insert(record);
		ASSERT_TRUE(result.ok);
		inserted_ids.push_back(new_id);
	}

	std::size_t resident_after = 0;
	for (RegionId region : index.allRegions()) {
		if (controller.acquireRegion(region).on_device) ++resident_after;
	}
	ARACHNE_LOG_INFO(
			"DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot: {} region(s) resident after insert (was {})",
			resident_after, resident_before);

	// Fresh host ground truth after the inserts -- should now surface the
	// near-duplicates (near-zero distance) at or near the top.
	TraverseResult after_insert_host = index.traverseHost({probe}).front();
	std::unordered_set<VectorId> host_found_inserted;
	for (const auto& n : after_insert_host.result.neighbors) {
		if (std::find(inserted_ids.begin(), inserted_ids.end(), n.id) != inserted_ids.end()) {
			host_found_inserted.insert(n.id);
		}
	}
	ARACHNE_LOG_INFO(
			"DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot: host ground truth after insert found {}/{} "
			"newly-inserted near-duplicates in top-{}",
			host_found_inserted.size(), inserted_ids.size(), probe.query.top_k);
	ASSERT_GT(host_found_inserted.size(), 0u)
			<< "test setup problem: even *host* search doesn't find the near-duplicates it just inserted";

	// The actual question: does traverseDevice() -- forced directly, not
	// through Controller's routing -- see the same thing, or does it still
	// serve whatever was resident before the insert?
	TraverseResult after_insert_device = index.traverseDevice({probe}).front();
	if (!after_insert_device.completed_within_scope) {
		ARACHNE_LOG_INFO(
				"DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot: traverseDevice() fell out of GPU-resident "
				"scope after the insert -- inconclusive for this run (some Region along the walk wasn't resident), not "
				"itself evidence of staleness");
		GTEST_SKIP() << "device walk fell out of scope this run -- see log for the resident region count before/after";
	}

	std::unordered_set<VectorId> device_found_inserted;
	for (const auto& n : after_insert_device.result.neighbors) {
		if (std::find(inserted_ids.begin(), inserted_ids.end(), n.id) != inserted_ids.end()) {
			device_found_inserted.insert(n.id);
		}
	}
	ARACHNE_LOG_INFO(
			"DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot: traverseDevice() after insert found {}/{} "
			"newly-inserted near-duplicates in top-{} (host found {})",
			device_found_inserted.size(), inserted_ids.size(), probe.query.top_k, host_found_inserted.size());

	// This is the actual finding this test is designed to surface: if this
	// fails, it means a host-side Insert into an already-promoted Region
	// left a stale device copy that traverseDevice() kept serving --
	// Controller::dispatch(const ModifyRequest&) has no on_complete hook to
	// invalidate/refresh residency after a completed Insert (unlike the
	// TraverseRequest overload's recordTraversal()/requestPromotion()), so
	// nothing in Core currently notices. If it passes, the mechanism this
	// port happens to rely on (engineLevel0Neighbors()/engineIsMarkedDeleted()
	// always reading *host* memory regardless of residency, and pre-existing
	// nodes' own vector bytes never being mutated in place by insert) is
	// what saves it in practice -- see hnsw_index_dist.hpp's own doc comment
	// on that host-read behavior.
	EXPECT_EQ(device_found_inserted.size(), host_found_inserted.size())
			<< "traverseDevice() and traverseHost() disagree on how many newly-inserted near-duplicates they found "
					 "after the insert -- see this test's own comment for what this would mean";
}

}  // namespace
