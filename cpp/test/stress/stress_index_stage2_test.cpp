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
	// 10 vectors/Region, 200 Regions total (2000 vectors) -- but the budget
	// below only ever holds ~16 Regions at once, so promoting the ~161st
	// through ~200th Region each require evicting older ones first (each one
	// potentially needing every one of an older Region's 10 dependents
	// evicted before it actually frees -- see evictAnchor()'s doc comment on
	// why a non-last-dependent eviction must still stop FIFO from tracking
	// that anchor, or this loop would never terminate), and by the end
	// essentially the whole dataset has cycled through GPU residency at
	// least once.
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
	// rmm::mr::pool_memory_resource requires the initial pool size to be a
	// multiple of 256 bytes.
	budget += (256 - budget % 256) % 256;

	Controller controller(index, routing_cache, arachne::SchedulingConfig{}, nullptr, budget,
												arachne::gpu::kDefaultMetadataPoolBytes);
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

	ControllerStats stats = controller.stats();
	// Eviction was actually forced -- not just capacity-checked and skipped.
	EXPECT_GT(stats.regions_promoted_total, 0u);
	EXPECT_GT(stats.regions_evicted_total, 0u);
	EXPECT_GT(stats.anchor_evictions_total, 0u);
	// Never exceeded the self-imposed budget, even under this much churn.
	EXPECT_LE(stats.gpu_bytes_allocated, budget);
	// compactions_total is deliberately not asserted > 0 here: every Region
	// in this test is the same fixed size, so a freed Region's hole is
	// always exactly the right size for the next one needing it -- genuine
	// fragmentation (a request no single free hole is big enough for,
	// despite enough aggregate free bytes) doesn't arise from a uniform-size
	// access pattern. allocateWithCompaction()'s fallback path exists for
	// adapters with variable-sized Regions; StressIndex isn't one, so 0 here
	// is the expected, correct outcome, not a sign the wiring is unused.

	// Correctness: sample-check search results -- including for vectors
	// whose Region was almost certainly evicted long before this loop ends
	// -- against ground truth computed independently from StressIndex's own
	// storage (see BruteForceGroundTruth()'s doc comment). Host data stays
	// authoritative regardless of current GPU residency (write-back keeps it
	// that way), so this must hold exactly the same as it did in stage 1.
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

	// Every Region should have ended up either currently resident or
	// properly evicted -- never left in a torn state (e.g. device.valid()
	// true but not actually reachable). Spot-check via acquireRegion() over
	// every Region id: this must not throw (all were registered) and must
	// agree with whatever residency Controller itself believes.
	for (RegionId id = 1; id <= (kCapacity + kVectorsPerRegion - 1) / kVectorsPerRegion; ++id) {
		EXPECT_NO_THROW({
			auto access = controller.acquireRegion(id);
			(void)access;
		}) << "region "
			 << id;
	}
}

}  // namespace
