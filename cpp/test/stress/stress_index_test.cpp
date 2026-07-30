// Stage 1 of the StressIndex plan: single-threaded, host-only (no forced
// eviction pressure -- capacity is sized to comfortably fit under the
// budget), correctness-only. Verifies Controller::insert()/search()/
// remove(), routed through a real ASRoutingCacheHnsw (not a stub), produce
// results matching BruteForceGroundTruth() -- i.e. that Arachne's whole
// orchestration pipeline (routing, promotion, scheduling, dispatch) doesn't
// lose or corrupt anything relative to what StressIndex actually stored.
// Runs across all four VectorDType values, dim=128, per the plan.

#include "stress_index.hpp"

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "gpu/device_context.hpp"
#include "gpu/dirty_header.hpp"
#include "stress_test_support.hpp"

namespace {

using arachne::ASRoutingCacheHnsw;
using arachne::Controller;
using arachne::ControllerStats;
using arachne::DeleteResult;
using arachne::DistanceMetric;
using arachne::InsertResult;
using arachne::Neighbor;
using arachne::Query;
using arachne::Record;
using arachne::SearchResult;
using arachne::VectorDType;
using arachne::VectorId;
using arachne::VectorView;
using arachne::stress::BruteForceGroundTruth;
using arachne::stress::StressIndex;
using arachne::stress::testsupport::GenerateVectors;

constexpr std::uint32_t kDim = 128;

class StressIndexStage1Test : public testing::TestWithParam<VectorDType> {};

TEST_P(StressIndexStage1Test, InsertThenSearchMatchesBruteForceGroundTruth) {
	VectorDType dtype = GetParam();
	constexpr std::size_t kCapacity = 2000;           // comfortably under the GPU budget below
	constexpr std::size_t kVectorsPerRegion = 64;
	constexpr std::size_t kNumVectors = 1500;
	constexpr std::uint32_t kTopK = 10;

	StressIndex index(kDim, dtype, kCapacity, kVectorsPerRegion);
	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/200, DistanceMetric::L2, dtype);

	// Budget covers the whole dataset comfortably -- stage 1 is about
	// correctness, not forcing eviction (that's stage 2).
	std::size_t element_size = arachne::VectorElementSize(dtype);
	std::size_t region_bytes = kVectorsPerRegion * kDim * element_size;
	std::size_t header_bytes = arachne::gpu::DirtyHeaderBytes(region_bytes, kDim * element_size);
	std::size_t num_regions = (kCapacity + kVectorsPerRegion - 1) / kVectorsPerRegion;
	std::size_t budget = (region_bytes + header_bytes) * num_regions;
	// rmm::mr::pool_memory_resource requires the initial pool size to be a
	// multiple of 256 bytes.
	budget += (256 - budget % 256) % 256;

	Controller controller(index, routing_cache, arachne::SchedulingConfig{}, nullptr, budget,
												 arachne::gpu::kDefaultMetadataPoolBytes);
	index.registerAllRegions(controller);

	std::mt19937 rng(42);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, kNumVectors, rng);

	for (std::size_t i = 0; i < kNumVectors; ++i) {
		Record record;
		record.id = static_cast<VectorId>(i) + 1;
		record.vector = VectorView{vectors[i].data(), kDim, dtype};
		InsertResult result = controller.insert(record);
		ASSERT_TRUE(result.ok) << "insert failed for vector " << i;
	}
	ASSERT_EQ(index.liveCount(), kNumVectors);

	// Search for a sample of the inserted vectors (not all -- O(N^2) full
	// scans add up) and confirm Controller::search() agrees with
	// BruteForceGroundTruth() computed independently against StressIndex's
	// own storage.
	constexpr std::size_t kNumQueries = 50;
	std::uniform_int_distribution<std::size_t> pick(0, kNumVectors - 1);
	for (std::size_t q = 0; q < kNumQueries; ++q) {
		std::size_t i = pick(rng);
		VectorView query_view{vectors[i].data(), kDim, dtype};

		Query query{query_view, kTopK};
		SearchResult searched = controller.search(query);
		std::vector<Neighbor> ground_truth = BruteForceGroundTruth(index, query_view, kTopK);

		ASSERT_EQ(searched.neighbors.size(), ground_truth.size()) << "query " << q << " (source vector " << i << ")";
		for (std::size_t k = 0; k < ground_truth.size(); ++k) {
			EXPECT_EQ(searched.neighbors[k].id, ground_truth[k].id) << "query " << q << " rank " << k;
			EXPECT_FLOAT_EQ(searched.neighbors[k].distance, ground_truth[k].distance) << "query " << q << " rank " << k;
		}
		// The query vector itself must always be its own nearest neighbor
		// (distance 0) for L2.
		ASSERT_FALSE(searched.neighbors.empty());
		EXPECT_EQ(searched.neighbors.front().id, static_cast<VectorId>(i) + 1) << "query " << q;
		EXPECT_FLOAT_EQ(searched.neighbors.front().distance, 0.0f) << "query " << q;
	}

	// Delete a handful and confirm they stop showing up in results.
	constexpr std::size_t kNumDeletes = 20;
	std::vector<VectorId> deleted_ids;
	for (std::size_t d = 0; d < kNumDeletes; ++d) {
		VectorId id = static_cast<VectorId>(d) + 1;  // first kNumDeletes inserted ids
		DeleteResult result = controller.remove(id);
		ASSERT_TRUE(result.ok) << "delete failed for id " << id;
		deleted_ids.push_back(id);
	}
	ASSERT_EQ(index.liveCount(), kNumVectors - kNumDeletes);

	for (VectorId id : deleted_ids) {
		std::size_t i = static_cast<std::size_t>(id) - 1;
		VectorView query_view{vectors[i].data(), kDim, dtype};
		Query query{query_view, kTopK};
		SearchResult searched = controller.search(query);
		for (const Neighbor& n : searched.neighbors) {
			EXPECT_NE(n.id, id) << "deleted id " << id << " still appeared in search results";
		}
	}

	// Promotion/eviction is now lazy (RegionManager's Coordinator) -- wait for
	// it to fully catch up before asserting on stats(), rather than relying
	// on however much of it happened to land during the insert/search/delete
	// loops above.
	controller.waitIdle();
	ControllerStats stats = controller.stats();
	EXPECT_GT(stats.regions_promoted_total, 0u);
	// The budget comfortably fits every registered Region, so *capacity*-
	// driven eviction (RegionManager::evictAnchorNow(), via
	// processPromotions()'s OutOfCapacity retry loop) should never fire here.
	// regions_evicted_total can still be nonzero though: it also counts a
	// Region freed because one of the kNumDeletes ids above happened to be
	// its sole remaining dependent at the moment it was deleted (see
	// RegionManager::releaseAnchor()) -- a legitimate, timing-dependent
	// consequence of deleting an early-inserted (and therefore often
	// sparsely-shared) Anchor, not eviction under pressure. Bounded by
	// kNumDeletes since at most one Region can be orphaned per delete.
	EXPECT_LE(stats.regions_evicted_total, kNumDeletes);
}

INSTANTIATE_TEST_SUITE_P(AllDTypes, StressIndexStage1Test,
													testing::Values(VectorDType::Int8, VectorDType::UInt8, VectorDType::Float16,
																					 VectorDType::Float32),
													[](const testing::TestParamInfo<VectorDType>& info) {
														switch (info.param) {
															case VectorDType::Int8:
																return "Int8";
															case VectorDType::UInt8:
																return "UInt8";
															case VectorDType::Float16:
																return "Float16";
															case VectorDType::Float32:
																return "Float32";
														}
														return "Unknown";
													});

}  // namespace
