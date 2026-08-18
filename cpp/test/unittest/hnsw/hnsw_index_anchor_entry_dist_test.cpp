// Correctness test for HnswIndexAnchorEntry's GPU path (index/hnsw/
// hnsw_index_anchor_entry.hpp/.cpp): unlike hnsw_index_dist_test.cpp (which
// exercises HnswIndexDist's *naive* traverseDevice() -- always the global
// entry point, beam width 1), this drives the real, end-to-end Controller
// flow *twice* per query so the entry-point cache actually gets a chance to
// populate and then be hit:
//
//   pass 1: controller.search(query) for a query never seen before -- no
//     RoutingCache hit yet, so this dispatches Hybrid -> traverseHost(),
//     now carrying the same anchor_id Controller's own requestPromotion()
//     uses (see controller.cpp's search()/insert() fix), letting
//     HnswIndexAnchorEntry cache that anchor_id -> the internal id its
//     search actually landed on.
//   waitIdle(): lets the Coordinator actually promote the touched Region(s)
//     and register the Anchor into RoutingCache (RegionManager only calls
//     RoutingCache::ensure() once a promotion actually succeeds).
//   pass 2: the *same* query again -- RoutingCache now hits (same
//     locality), the matched Anchor's Region is already resident, so this
//     dispatches GpuOnly -> traverseDevice() -> resolveEntryPoint() should
//     hit the cache populated in pass 1.
//
// Accuracy bar: this class's search is intentionally approximate relative
// to hnswlib itself (no upper-level descent, beam width > 1 explores some
// candidates a strict best-first walk wouldn't have -- see
// report-hnsw-dist.md), so this does not require exact agreement with
// traverseHost()'s ground truth. Instead it measures top-k overlap between
// the two and asserts it's well above chance -- proof the graph walk is
// actually finding real neighbors, not asserting hnswlib-identical recall.

#include "hnsw_index_anchor_entry.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "stress_test_support.hpp"

namespace {

using arachne::ASRoutingCacheHnsw;
using arachne::Controller;
using arachne::DistanceMetric;
using arachne::Query;
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
constexpr std::size_t kCapacity = 2200;
constexpr std::size_t kVectorsPerRegion = 32;
constexpr std::size_t kM = 16;
constexpr std::size_t kEfConstruction = 100;
constexpr std::size_t kNumVectors = 2000;
constexpr std::uint32_t kTopK = 5;

std::size_t Overlap(const TraverseResult& a, const TraverseResult& b) {
	std::unordered_set<VectorId> ids;
	for (const auto& n : a.result.neighbors) ids.insert(n.id);
	std::size_t overlap = 0;
	for (const auto& n : b.result.neighbors) {
		if (ids.count(n.id)) ++overlap;
	}
	return overlap;
}

TEST(HnswIndexAnchorEntryDistTest, TraverseDeviceUsesCachedAnchorEntryPointAndStaysApproximatelyAccurate) {
	std::mt19937 rng(31);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

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
	Controller controller(index, routing_cache, SchedulingConfig{}, nullptr, total_host_bytes * 2);
	index.registerAllRegions(controller);
	index.attachController(controller);

	// Query indices spread across the dataset -- every 30th vector, self-query
	// (querying with a point's own stored vector), so ground truth via
	// traverseHost() is cheap to sanity-check independently.
	std::vector<std::size_t> query_indices;
	for (std::size_t i = 0; i < vectors.size(); i += 30) query_indices.push_back(i);

	// Pass 1: first-ever encounter of each query -- Hybrid, populates the
	// per-Anchor entry-point cache (see this file's own overview). waitIdle()
	// after *every* query rather than once at the end: with only ~63 Regions
	// covering the whole dataset, firing all 67 never-before-seen queries back
	// to back lets most of their freshly-minted Anchors land in the very same
	// Coordinator batch and contend over the same handful of Regions --
	// RegionManager::make() defers (rather than retries within the same pass)
	// an Anchor that loses that race, so most promotions never actually
	// publish before waitIdle() returns. Settling between queries keeps each
	// promotion uncontested, closer to how distinct queries would actually
	// arrive over time anyway.
	for (std::size_t i : query_indices) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, kTopK};
		SearchResult unused = controller.search(query);
		(void)unused;
		controller.waitIdle();
	}

	std::size_t resident_regions = 0;
	for (RegionId region : index.allRegions()) {
		if (controller.acquireRegion(region).on_device) ++resident_regions;
	}
	ASSERT_GT(resident_regions, 0u) << "no Region got promoted after pass 1 -- test setup itself is broken";

	// Pass 2: same queries again -- should now route GpuOnly for at least
	// some of them, through HnswIndexAnchorEntry's cached entry point.
	std::size_t served_gpu_only = 0;
	std::size_t compared = 0;
	std::size_t total_overlap = 0;
	std::size_t total_neighbors_compared = 0;
	for (std::size_t i : query_indices) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, kTopK};
		SearchResult second = controller.search(query);
		if (second.served_gpu_only) ++served_gpu_only;

		// Independent ground truth: hnswlib's own search via traverseHost(),
		// bypassing Controller entirely.
		TraverseRequest host_request;
		host_request.query = query;
		TraverseResult host_result = index.traverseHost({host_request}).front();

		TraverseResult device_shaped;
		device_shaped.result = second;
		std::size_t overlap = Overlap(host_result, device_shaped);
		total_overlap += overlap;
		total_neighbors_compared += host_result.result.neighbors.size();
		++compared;
	}
	controller.waitIdle();

	ASSERT_GT(served_gpu_only, 0u) << "no query in pass 2 was served GpuOnly out of " << compared
																	<< " -- the anchor-cache/promotion pipeline never actually engaged the "
																		 "device path, so nothing GPU-specific was exercised";
	double overlap_ratio = total_neighbors_compared > 0
														 ? static_cast<double>(total_overlap) / static_cast<double>(total_neighbors_compared)
														 : 0.0;
	// Chance-level overlap for top-5 out of ~2000 candidates is close to zero;
	// this bar (well above chance, comfortably below "must match exactly") is
	// the "정확도가 아주 일치하지는 않더라도 비슷한 무언가" bar this test is meant to prove.
	EXPECT_GE(overlap_ratio, 0.3) << "average top-" << kTopK
																 << " overlap between Controller::search() and ground-truth traverseHost() "
																		"was only "
																 << overlap_ratio << " across " << compared << " queries -- too low to call this "
																 << "'approximately accurate'";

	// Direct, deterministic check of the cache mechanism itself (isolated from
	// Controller's probabilistic promotion timing): call traverseHost() with
	// an explicit, never-before-seen anchor_id to populate the cache from
	// this exact query, then traverseDevice() with the *same* anchor_id and
	// same query. If the cache is wired correctly, the walk starts right at
	// (or immediately next to) the node this exact query already converged
	// on, so it should reliably reproduce a similar answer -- unlike the
	// aggregate pass-2 check above (which mixes in Controller's own
	// promotion/eviction timing), this isolates HnswIndexAnchorEntry's own
	// resolveEntryPoint()/traverseHost() contract from everything else.
	{
		for (std::size_t i : query_indices) {
			TraverseRequest seed_request;
			seed_request.query.vector = VectorView{vectors[i].data(), kDim, VectorDType::Float32};
			seed_request.query.top_k = kTopK;
			seed_request.anchor_id = VectorId{1'000'000 + i};  // guaranteed never cached before this call
			TraverseResult seeded = index.traverseHost({seed_request}).front();
			ASSERT_FALSE(seeded.result.neighbors.empty());

			TraverseResult device_result = index.traverseDevice({seed_request}).front();
			if (!device_result.completed_within_scope) continue;  // some Region along the way wasn't resident -- skip
			ASSERT_FALSE(device_result.result.neighbors.empty());
			EXPECT_GT(Overlap(seeded, device_result), 0u)
					<< "query index " << i << ": traverseDevice() using the just-cached entry point for a "
					<< "brand-new anchor_id shares zero top-" << kTopK << " results with the host search that "
					<< "populated its cache -- the cached entry point likely isn't being used correctly";
		}
	}
}

}  // namespace
