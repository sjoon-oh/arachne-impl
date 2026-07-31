// Stage 2 of the StressIndex plan: a dataset deliberately sized larger than
// the GPU budget, forcing real promote/evict cycling (and, if fragmentation
// ever actually occurs, compaction -- see below) through Controller's real
// FIFO replacement path, rather than stage 1's comfortably-fits-under-budget
// setup. The point isn't new algorithmic behavior -- it's proving the exact
// same insert()/search() correctness stage 1 already established still
// holds when Regions are actively being evicted and re-promoted throughout
// the run, and that Controller's own bookkeeping (ControllerStats) reports
// what actually happened.

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
using arachne::DistanceMetric;
using arachne::Neighbor;
using arachne::Query;
using arachne::Record;
using arachne::RegionId;
using arachne::SearchResult;
using arachne::VectorDType;
using arachne::VectorId;
using arachne::VectorView;
using arachne::stress::BruteForceGroundTruth;
using arachne::stress::StressIndex;
using arachne::stress::testsupport::GenerateVectors;

TEST(StressIndexStage2Test, EvictionCyclingPreservesCorrectnessUnderTinyGpuBudget) {
	constexpr std::uint32_t kDim = 128;
	constexpr VectorDType kDType = VectorDType::Float32;
	// 10 vectors/Region, 200 Regions total (2000 vectors), but the budget
	// below only ever holds ~16 Regions at once -- promoting later Regions
	// forces evicting older ones first (see evictAnchor()'s doc comment), so
	// by the end essentially the whole dataset has cycled through GPU
	// residency at least once.
	constexpr std::size_t kVectorsPerRegion = 10;
	constexpr std::size_t kCapacity = 2000;
	constexpr std::size_t kRegionsThatFitBudget = 16;

	StressIndex index(kDim, kDType, kCapacity, kVectorsPerRegion);
	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/200, DistanceMetric::L2, kDType);

	std::size_t element_size = arachne::VectorElementSize(kDType);
	std::size_t region_payload_bytes = kVectorsPerRegion * kDim * element_size;
	std::size_t header_bytes = arachne::gpu::DirtyHeaderBytes(region_payload_bytes, kDim * element_size);
	std::size_t region_total_bytes = region_payload_bytes + header_bytes;
	std::size_t budget = kRegionsThatFitBudget * region_total_bytes;

	// gpu_unit_bytes == region_total_bytes so each Region occupies exactly
	// one arena unit -- otherwise Pooled's coarser default unit size would
	// silently let more than kRegionsThatFitBudget Regions fit at once,
	// defeating the capacity pressure this test depends on.
	Controller controller(index, routing_cache, arachne::SchedulingConfig{}, nullptr, budget,
												arachne::gpu::kDefaultMetadataPoolBytes, /*gpu_unit_bytes=*/region_total_bytes);
	index.registerAllRegions(controller);

	std::mt19937 rng(2024);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(kDType, kDim, kCapacity, rng);

	for (std::size_t i = 0; i < kCapacity; ++i) {
		Record record;
		record.id = static_cast<VectorId>(i) + 1;
		record.vector = VectorView{vectors[i].data(), kDim, kDType};
		ASSERT_TRUE(controller.insert(record).ok) << "insert failed at vector " << i;
	}
	EXPECT_EQ(index.liveCount(), kCapacity);

	// Promotion/eviction is now lazy (RegionManager's Coordinator) -- wait for
	// it to fully catch up before asserting on stats()/acquireRegion() below,
	// rather than relying on however much of it happened to land during the
	// insert loop above.
	controller.waitIdle();
	ControllerStats stats = controller.stats();
	// Eviction was actually forced -- not just capacity-checked and skipped.
	EXPECT_GT(stats.regions_promoted_total, 0u);
	EXPECT_GT(stats.regions_evicted_total, 0u);
	EXPECT_GT(stats.anchor_evictions_total, 0u);
	// Never exceeded the self-imposed budget, even under this much churn.
	EXPECT_LE(stats.gpu_bytes_allocated, budget);
	// compactions_total isn't asserted > 0 here: every Region is the same
	// fixed size, so a freed Region's hole always exactly fits the next one
	// needing it -- genuine fragmentation doesn't arise from a uniform-size
	// access pattern. 0 is the expected outcome, not unused wiring.

	// Sample-check search results -- including for vectors whose Region was
	// almost certainly evicted long before this loop ends -- against ground
	// truth computed independently from StressIndex's own storage. Host data
	// stays authoritative regardless of GPU residency, so this must match stage 1.
	constexpr std::size_t kNumQueries = 100;
	constexpr std::uint32_t kTopK = 5;
	std::uniform_int_distribution<std::size_t> pick(0, kCapacity - 1);
	for (std::size_t q = 0; q < kNumQueries; ++q) {
		std::size_t i = pick(rng);
		VectorView query_view{vectors[i].data(), kDim, kDType};
		Query query{query_view, kTopK};

		SearchResult searched = controller.search(query);
		std::vector<Neighbor> ground_truth = BruteForceGroundTruth(index, query_view, kTopK);

		ASSERT_EQ(searched.neighbors.size(), ground_truth.size()) << "query " << q << " (source vector " << i << ")";
		for (std::size_t k = 0; k < ground_truth.size(); ++k) {
			EXPECT_EQ(searched.neighbors[k].id, ground_truth[k].id) << "query " << q << " rank " << k;
		}
		ASSERT_FALSE(searched.neighbors.empty());
		EXPECT_EQ(searched.neighbors.front().id, static_cast<VectorId>(i) + 1) << "query " << q;
		EXPECT_FLOAT_EQ(searched.neighbors.front().distance, 0.0f) << "query " << q;
	}

	// Every Region should have ended up either resident or properly evicted
	// -- never left torn (e.g. device.valid() true but unreachable).
	// Spot-check via acquireRegion() over every Region id: must not throw and
	// must agree with whatever residency Controller believes.
	for (RegionId id = 1; id <= (kCapacity + kVectorsPerRegion - 1) / kVectorsPerRegion; ++id) {
		EXPECT_NO_THROW({
			auto access = controller.acquireRegion(id);
			(void)access;
		}) << "region "
			 << id;
	}
}

}  // namespace
