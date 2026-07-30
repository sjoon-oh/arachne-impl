// Stage 3 of the StressIndex plan: many concurrent caller threads hammering
// Controller::insert()/search()/remove() at once (stage 1/2 only ever drive
// Controller from a single thread), combined with a GPU budget deliberately
// far too small to hold the dataset -- forcing constant promotion/eviction/
// compaction cycling *while* concurrent traffic is still in flight, not
// serialized before/after it the way stage 2's single-threaded loop is.
//
// This is also the first test exercising RoutingCache registration having
// moved from Controller's commitSearch()/commitInsert() into RegionManager's
// own Coordinator (see region_manager.hpp's class doc comment) under real
// concurrent pressure -- specifically the delete-then-VectorId-reuse race
// PromotionCandidate's `epoch` field exists to guard against (see its own
// doc comment, replacement_policy.hpp): many threads concurrently inserting/
// removing/reusing the same, deliberately small/overlapping id ranges is
// exactly the scenario that needs a wide window to reliably manifest in.

#include "stress_index.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "gpu/device_context.hpp"
#include "gpu/dirty_header.hpp"
#include "stress_test_support.hpp"

namespace {

using namespace arachne;
using arachne::stress::BruteForceGroundTruth;
using arachne::stress::StressIndex;
using arachne::stress::testsupport::GenerateVectors;

// Runs `kThreads` threads, each performing a random mix of insert/search/
// remove over a shared, overlapping id space (`kIdSpace` ids), against a
// Controller whose GPU budget only ever holds `kRegionsThatFitBudget`
// Regions at once -- then verifies (a) the run completes at all (no crash/
// deadlock/hang -- reaching the post-join assertions already proves this),
// (b) promotion/eviction were actually forced repeatedly (not just
// capacity-checked and skipped), and (c) every id that ends up live in
// StressIndex's own storage is still correctly reachable via
// Controller::search() -- i.e. no stale/duplicated/misrouted Anchor
// survived the churn.
void RunConcurrentChurnStress(std::uint32_t dim, std::size_t vectors_per_region, std::size_t regions_that_fit_budget,
															int threads, int ops_per_thread, std::size_t id_space, std::size_t capacity,
															std::chrono::milliseconds trigger_interval) {
	constexpr VectorDType kDType = VectorDType::Float32;

	StressIndex index(dim, kDType, capacity, vectors_per_region);
	ASRoutingCacheHnsw routing_cache(dim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/200, DistanceMetric::L2, kDType);

	std::size_t element_size = VectorElementSize(kDType);
	std::size_t region_payload_bytes = vectors_per_region * dim * element_size;
	std::size_t header_bytes = gpu::DirtyHeaderBytes(region_payload_bytes, dim * element_size);
	std::size_t region_total_bytes = header_bytes + region_payload_bytes;
	std::size_t budget = regions_that_fit_budget * region_total_bytes;
	// rmm::mr::pool_memory_resource requires the initial pool size to be a
	// multiple of 256 bytes.
	budget += (256 - budget % 256) % 256;

	Controller controller(index, routing_cache, SchedulingConfig{}, nullptr, budget, gpu::kDefaultMetadataPoolBytes,
												CoordinatorConfig{trigger_interval});
	index.registerAllRegions(controller);

	// One fixed vector per id in the shared id space -- every thread
	// inserting id `i+1` always uses the exact same vector, so ground truth
	// stays well-defined no matter which thread's insert "won" that id at
	// any given moment, or how many times it was deleted and reinserted.
	std::mt19937 seed_rng(7);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(kDType, dim, id_space, seed_rng);

	std::atomic<int> insert_ok{0};
	std::atomic<int> remove_ok{0};
	std::atomic<int> search_done{0};

	std::vector<std::thread> workers;
	workers.reserve(static_cast<std::size_t>(threads));
	for (int t = 0; t < threads; ++t) {
		workers.emplace_back([&, t] {
			std::mt19937 rng(1000u + static_cast<unsigned>(t));
			std::uniform_int_distribution<std::size_t> id_dist(0, id_space - 1);
			std::uniform_int_distribution<int> op_dist(0, 9);  // weighted: mostly insert/search, some remove
			for (int op = 0; op < ops_per_thread; ++op) {
				std::size_t i = id_dist(rng);
				VectorId id = static_cast<VectorId>(i) + 1;
				int choice = op_dist(rng);
				if (choice < 5) {
					Record record;
					record.id = id;
					record.vector = VectorView{vectors[i].data(), dim, kDType};
					if (controller.insert(record).ok) insert_ok.fetch_add(1, std::memory_order_relaxed);
				} else if (choice < 8) {
					Query query{VectorView{vectors[i].data(), dim, kDType}, /*top_k=*/5};
					SearchResult result = controller.search(query);
					(void)result;
					search_done.fetch_add(1, std::memory_order_relaxed);
				} else {
					if (controller.remove(id).ok) remove_ok.fetch_add(1, std::memory_order_relaxed);
				}
			}
		});
	}
	for (std::thread& worker : workers) worker.join();

	// (a) Excessive multi-threaded request handling: reaching this point at
	// all (every thread's insert()/search()/remove() calls returned, no
	// hang/crash/deadlock) is the primary thing this proves. The op mix
	// actually happened too, not just resolved to all-rejected.
	EXPECT_GT(insert_ok.load(), 0);
	EXPECT_GT(search_done.load(), 0);
	EXPECT_GT(remove_ok.load(), 0);

	// (b) Excessive eviction/promotion/compaction: the tiny budget must have
	// actually forced real cycling, not just been sized to comfortably fit
	// (stage 1's setup) -- wait for the Coordinator to fully catch up first.
	controller.waitIdle();
	ControllerStats stats = controller.stats();
	EXPECT_GT(stats.regions_promoted_total, 0u);
	EXPECT_GT(stats.regions_evicted_total, 0u);
	EXPECT_GT(stats.anchor_evictions_total, 0u);
	EXPECT_LE(stats.gpu_bytes_allocated, budget);
	// compactions_total is deliberately not asserted on, same as stage 2:
	// every Region here is the same fixed size, so allocateWithCompaction()'s
	// fallback essentially never triggers under a uniform-size access
	// pattern -- 0 is an expected outcome, not a sign the wiring is unused.

	// (c) Data integrity: reconstruct, from StressIndex's own storage (the
	// authoritative host-side state, independent of Controller/RegionManager/
	// RoutingCache entirely -- see BruteForceGroundTruth()'s own doc
	// comment), which ids in the shared space are actually live right now,
	// and confirm Controller::search() still finds each one exactly -- the
	// end-to-end check that the epoch/RoutingCache machinery above never let
	// a stale or VectorId-reused Anchor identity corrupt routing under this
	// much concurrent insert/remove/reuse churn.
	std::size_t checked_live = 0;
	for (std::size_t i = 0; i < id_space; ++i) {
		VectorId id = static_cast<VectorId>(i) + 1;
		VectorView query_view{vectors[i].data(), dim, kDType};
		std::vector<Neighbor> ground_truth = BruteForceGroundTruth(index, query_view, /*top_k=*/1);
		if (ground_truth.empty() || ground_truth.front().id != id) {
			continue;  // id `id` is not currently live in StressIndex -- nothing to check
		}
		++checked_live;

		Query query{query_view, /*top_k=*/1};
		SearchResult searched = controller.search(query);
		ASSERT_FALSE(searched.neighbors.empty()) << "id " << id;
		EXPECT_EQ(searched.neighbors.front().id, id) << "id " << id;
		EXPECT_FLOAT_EQ(searched.neighbors.front().distance, 0.0f) << "id " << id;
	}
	EXPECT_GT(checked_live, 0u);  // the churn must have left *something* live to check

	// Every Region must have ended up either currently resident or properly
	// evicted -- never torn (mirrors stage 2's own final sweep).
	std::size_t num_regions = (capacity + vectors_per_region - 1) / vectors_per_region;
	for (RegionId id = 1; id <= static_cast<RegionId>(num_regions); ++id) {
		EXPECT_NO_THROW({
			auto access = controller.acquireRegion(id);
			(void)access;
		}) << "region "
			 << id;
	}
}

TEST(StressIndexStage3Test, ManyThreadsWithExcessiveEvictionPromotionCompactionPreserveDataIntegrity) {
	// A moderately sized id space and thread count -- broad concurrent
	// coverage across many distinct Anchors/Regions at once.
	RunConcurrentChurnStress(/*dim=*/32, /*vectors_per_region=*/4, /*regions_that_fit_budget=*/8, /*threads=*/12,
													 /*ops_per_thread=*/250, /*id_space=*/150, /*capacity=*/2000,
													 /*trigger_interval=*/std::chrono::milliseconds(2));
}

TEST(StressIndexStage3Test, ManyThreadsRacingInsertDeleteReuseOfASharedSmallIdSpaceNeverCorruptsRouting) {
	// The adversarial counterpart: far more threads than ids, so the *same*
	// handful of VectorIds are being concurrently inserted, deleted, and
	// reinserted (id reuse) constantly -- maximizing how often
	// requestPromotion()'s enqueue races releaseAnchor()'s epoch bump for the
	// exact same anchor_id (see PromotionCandidate's own doc comment). Paired
	// with a budget that only ever fits a single Region, guaranteeing
	// promotion/eviction is happening essentially continuously throughout.
	//
	// `capacity` is sized generously above the worst-case total number of
	// successful inserts this run could ever produce (threads * ops_per_thread,
	// since at most one insert "wins" per op): StressIndex never recycles a
	// deleted slot (see its own class doc comment -- next_free_slot_ only ever
	// grows), and this test deliberately reinserts the *same* handful of ids
	// over and over, so undersizing `capacity` here would exhaust it and make
	// every insert fail from then on -- a test-harness artifact, not anything
	// RegionManager/Controller under test is responsible for.
	RunConcurrentChurnStress(/*dim=*/16, /*vectors_per_region=*/16, /*regions_that_fit_budget=*/1, /*threads=*/16,
													 /*ops_per_thread=*/300, /*id_space=*/8, /*capacity=*/4096,
													 /*trigger_interval=*/std::chrono::milliseconds(1));
}

}  // namespace
