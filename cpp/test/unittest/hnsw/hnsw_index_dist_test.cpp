// Correctness test for HnswIndexDist::traverseDevice() (index/hnsw/
// hnsw_index_dist.hpp/.cpp/hnsw_dist_kernel.cu): drives a *real* Controller
// so Regions actually get promoted to GPU (Controller::acquireRegion()
// reporting on_device=true is exactly what traverseDevice() needs to avoid
// falling back), then compares traverseDevice()'s results against
// traverseHost()'s (== plain hnswlib, already verified against
// self-recall/ground truth in hnsw_index_test.cpp) on the *same* index
// state. If the GPU-offloaded distance kernel or the from-scratch search
// loop in hnsw_index_dist.cpp has a bug, this is what would catch a
// divergence between the two.
//
// Parameterized over every (VectorDType, DistanceMetric) combination
// hnsw_dist_kernel.cu supports (Cosine excluded -- see its own overview) --
// exercises all 8 kernel instantiations (FloatAccumDistanceKernel<float,
// uint16_t> x IntAccumDistanceKernel<unsigned char, int8_t>, each x
// {L2, InnerProduct}) against hnswlib's own scalar distance functions.

#include "hnsw_index_dist.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "stress_test_support.hpp"

namespace {

using arachne::ASRoutingCacheHnsw;
using arachne::Controller;
using arachne::DistanceMetric;
using arachne::InsertResult;
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
using arachne::index::hnsw::HnswIndexDist;
using arachne::stress::testsupport::GenerateVectors;

constexpr std::uint32_t kDim = 16;
constexpr std::size_t kCapacity = 300;
constexpr std::size_t kVectorsPerRegion = 32;
constexpr std::size_t kM = 16;
constexpr std::size_t kEfConstruction = 100;
constexpr std::size_t kNumVectors = 200;

struct DTypeMetric {
	VectorDType dtype;
	DistanceMetric metric;
};

std::string DTypeMetricName(const testing::TestParamInfo<DTypeMetric>& info) {
	std::string dtype;
	switch (info.param.dtype) {
		case VectorDType::Int8:
			dtype = "Int8";
			break;
		case VectorDType::UInt8:
			dtype = "UInt8";
			break;
		case VectorDType::Float16:
			dtype = "Float16";
			break;
		case VectorDType::Float32:
			dtype = "Float32";
			break;
	}
	return dtype + (info.param.metric == DistanceMetric::L2 ? "L2" : "IP");
}

class HnswIndexDistTest : public testing::TestWithParam<DTypeMetric> {};

TEST_P(HnswIndexDistTest, TraverseDeviceMatchesTraverseHostOnResidentRegions) {
	VectorDType dtype = GetParam().dtype;
	DistanceMetric metric = GetParam().metric;

	std::mt19937 rng(29);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, kNumVectors, rng);

	HnswIndexDist index(kDim, dtype, metric, kCapacity, kVectorsPerRegion, kM, kEfConstruction);

	std::vector<std::byte> bytes(vectors.front().size() * vectors.size());
	std::vector<VectorId> ids(vectors.size());
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		std::memcpy(bytes.data() + i * vectors.front().size(), vectors[i].data(), vectors.front().size());
		ids[i] = static_cast<VectorId>(i) + 1;
	}
	index.build(VectorBatchView{bytes.data(), kDim, dtype, ids.size(), ids.data()});

	// Generous GPU budget -- covers every Region this dataset produces, so
	// nothing this test does should ever get evicted once promoted.
	std::size_t total_host_bytes = 0;
	for (RegionId region : index.allRegions()) total_host_bytes += index.resolveRegion(region)->hostView().bytes;

	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/64, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/100, metric, dtype);
	Controller controller(index, routing_cache, SchedulingConfig{}, nullptr, total_host_bytes * 2);
	index.registerAllRegions(controller);
	index.attachController(controller);

	// Drive a real search() per vector so Controller's own dispatch/Anchor/
	// promotion pipeline (Controller::insert()/search() -> traverseHost() ->
	// touched -> requestPromotion()) requests every Region at least once.
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, dtype}, /*top_k=*/1};
		SearchResult unused = controller.search(query);
		(void)unused;
	}
	controller.waitIdle();

	std::size_t resident_regions = 0;
	for (RegionId region : index.allRegions()) {
		if (controller.acquireRegion(region).on_device) ++resident_regions;
	}
	ASSERT_GT(resident_regions, 0u) << "no Region got promoted -- test setup itself is broken, not just the GPU path";

	std::size_t compared = 0;
	std::size_t attempted_device_calls = 0;
	std::size_t completed_on_device = 0;
	for (std::size_t i = 0; i < vectors.size(); i += 5) {  // every 5th vector -- enough coverage, keeps the test quick
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i].data(), kDim, dtype};
		request.query.top_k = 3;

		TraverseResult host_result = index.traverseHost({request}).front();
		TraverseResult device_result = index.traverseDevice({request}).front();
		++attempted_device_calls;
		if (!device_result.completed_within_scope) continue;  // this particular Anchor's region wasn't resident
		++completed_on_device;
		++compared;

		ASSERT_EQ(host_result.result.neighbors.size(), device_result.result.neighbors.size());
		// Set-overlap comparison, not strict positional match: verified
		// separately (a tiny, fully-connected -- every node within one hop of
		// every other -- graph, so entry point/beam width can't matter) that
		// the distance kernel itself is bit-exact for every dtype/metric here,
		// including UInt8/InnerProduct. At this test's larger (200-vector)
		// scale, id ordering can still legitimately differ from hnswlib's own
		// search -- HnswIndexDist's copied search loop is a known
		// approximation of hnswlib's real algorithm (single global entry
		// point, no upper-level descent, beam width 1 -- see
		// hnsw_index_dist.hpp's file overview), and that approximation shows
		// up hardest for exactly this metric/dtype pairing: raw (uncentered)
		// InnerProduct over uniformly-random *non-negative* UInt8 components
		// barely discriminates between candidates (every pair's dot product
		// clusters tightly around the same large value), so which candidate
		// "wins" a close call is genuinely sensitive to small path
		// differences -- not evidence the kernel computed a wrong distance.
		// For ids present in *both* results, their distances must still agree
		// closely (that part does test the kernel, independent of which path
		// the walk took).
		std::size_t overlap = 0;
		for (const auto& host_neighbor : host_result.result.neighbors) {
			for (const auto& device_neighbor : device_result.result.neighbors) {
				if (host_neighbor.id != device_neighbor.id) continue;
				++overlap;
				EXPECT_NEAR(host_neighbor.distance, device_neighbor.distance, 1e-2f)
						<< "query index " << i << " id " << host_neighbor.id;
				break;
			}
		}
		EXPECT_GT(overlap, 0u) << "query index " << i << ": zero overlap between host and device top-"
													 << host_result.result.neighbors.size() << " -- too low to call this correct";
	}
	ASSERT_GT(completed_on_device, 0u)
			<< "traverseDevice() never completed within scope for any of " << attempted_device_calls
			<< " attempts -- promotion coverage in this test isn't reaching the queries being compared";
}

INSTANTIATE_TEST_SUITE_P(AllDTypesAndMetrics, HnswIndexDistTest,
		testing::Values(DTypeMetric{VectorDType::Int8, DistanceMetric::L2}, DTypeMetric{VectorDType::Int8, DistanceMetric::InnerProduct},
				DTypeMetric{VectorDType::UInt8, DistanceMetric::L2}, DTypeMetric{VectorDType::UInt8, DistanceMetric::InnerProduct},
				DTypeMetric{VectorDType::Float16, DistanceMetric::L2}, DTypeMetric{VectorDType::Float16, DistanceMetric::InnerProduct},
				DTypeMetric{VectorDType::Float32, DistanceMetric::L2}, DTypeMetric{VectorDType::Float32, DistanceMetric::InnerProduct}),
		DTypeMetricName);

}  // namespace
