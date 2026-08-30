// Correctness test for HnswlibIndexGpu::traverseDevice() (index/hnsw/
// hnswlib_index_gpu.hpp/.cpp/hnsw_dist_kernel.cu): drives a *real* Controller
// so Regions actually get promoted to GPU (Controller::acquireRegion()
// reporting on_device=true is exactly what traverseDevice() needs to avoid
// falling back), then compares traverseDevice()'s results against
// traverseHost()'s (== plain hnswlib, already verified against
// self-recall/ground truth in hnswlib_index_test.cpp) on the *same* index
// state. If the GPU-offloaded distance kernel or the from-scratch search
// loop in hnswlib_index_gpu.cpp has a bug, this is what would catch a
// divergence between the two.
//
// Parameterized over every (VectorDType, DistanceMetric) combination
// hnsw_dist_kernel.cu supports (Cosine excluded -- see its own overview) --
// exercises all 8 kernel instantiations (FloatAccumDistanceKernel<float,
// uint16_t> x IntAccumDistanceKernel<unsigned char, int8_t>, each x
// {L2, InnerProduct}) against hnswlib's own scalar distance functions.

#include "hnswlib_index_gpu.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
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
using arachne::index::hnsw::HnswlibIndexGpu;
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

class HnswlibIndexGpuTest : public testing::TestWithParam<DTypeMetric> {};

TEST_P(HnswlibIndexGpuTest, TraverseDeviceMatchesTraverseHostOnResidentRegions) {
	VectorDType dtype = GetParam().dtype;
	DistanceMetric metric = GetParam().metric;

	std::mt19937 rng(29);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, kNumVectors, rng);

	HnswlibIndexGpu index(kDim, dtype, metric, kCapacity, kVectorsPerRegion, kM, kEfConstruction);

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
		// completed_within_scope is unconditionally true now -- a candidate
		// whose Region isn't GPU-resident is computed on host instead of
		// aborting the walk (see hnswlib_index_gpu.hpp's own doc comment on
		// TraverseBatchOnDevice()/compute_distances_batch()), so there's no longer a
		// "didn't complete within scope" case to skip here.
		++completed_on_device;
		++compared;

		ASSERT_EQ(host_result.result.neighbors.size(), device_result.result.neighbors.size());
		// Set-overlap comparison, not strict positional match: verified
		// separately (a tiny, fully-connected -- every node within one hop of
		// every other -- graph, so entry point/beam width can't matter) that
		// the distance kernel itself is bit-exact for every dtype/metric here,
		// including UInt8/InnerProduct. At this test's larger (200-vector)
		// scale, id ordering can still legitimately differ from hnswlib's own
		// search -- HnswlibIndexGpu's copied search loop is a known
		// approximation of hnswlib's real algorithm (single global entry
		// point, no upper-level descent, beam width 1 -- see
		// hnswlib_index_gpu.hpp's file overview), and that approximation shows
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

// Verifies the partial-residency redesign of TraverseBatchOnDevice()/
// compute_distances_batch() (hnswlib_index_gpu.cpp): a deliberately small GPU budget
// forces some Regions resident and others not, so every query below is
// guaranteed to hit a *mix* of GPU- and host-computed candidates within a
// single walk -- the exact case the old all-or-nothing design would have
// bailed on entirely (completed_within_scope=false). Accuracy is judged the
// same way as the fully-resident parameterized test above: set-overlap
// against traverseHost() ground truth, plus close distance agreement for
// ids present in both.
TEST(HnswlibIndexGpuPartialResidencyTest, StaysAccurateWithAMixOfResidentAndNonResidentRegions) {
	std::mt19937 rng(41);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

	HnswlibIndexGpu index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
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
	// About a third of the dataset -- small enough that promotion pressure
	// across every vector below can't fit it all, generous enough that
	// *some* Regions do get promoted (a genuine mix, not zero residency --
	// see the zero-residency test below for that separate case).
	Controller controller(index, routing_cache, SchedulingConfig{}, nullptr, total_host_bytes / 3);
	index.registerAllRegions(controller);
	index.attachController(controller);

	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, /*top_k=*/1};
		controller.search(query);
	}
	controller.waitIdle();

	std::size_t resident = 0;
	std::size_t total_regions = 0;
	for (RegionId region : index.allRegions()) {
		++total_regions;
		if (controller.acquireRegion(region).on_device) ++resident;
	}
	ASSERT_GT(resident, 0u) << "test setup problem: nothing got promoted at all";
	ASSERT_LT(resident, total_regions) << "test setup problem: everything got promoted -- budget wasn't actually "
																				 "constraining, this isn't testing partial residency";

	std::size_t compared = 0;
	for (std::size_t i = 0; i < vectors.size(); i += 5) {
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i].data(), kDim, VectorDType::Float32};
		request.query.top_k = 3;

		TraverseResult host_result = index.traverseHost({request}).front();
		TraverseResult device_result = index.traverseDevice({request}).front();
		EXPECT_TRUE(device_result.completed_within_scope);
		ASSERT_EQ(host_result.result.neighbors.size(), device_result.result.neighbors.size());

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
		EXPECT_GT(overlap, 0u) << "query index " << i << ": zero overlap under partial residency";
		++compared;
	}
	ASSERT_GT(compared, 0u);
}

// The extreme end of the same redesign: *zero* Regions ever promoted (no
// warm-up searches at all, unlike every other test in this file), so every
// single candidate in every walk is computed on host. Confirms
// traverseDevice() is now a fully self-sufficient traversal even with no GPU
// help whatsoever -- no more Controller-level Hybrid retry needed for this
// case, since there's no more failure mode to retry from.
TEST(HnswlibIndexGpuPartialResidencyTest, StaysAccurateWithZeroResidency) {
	std::mt19937 rng(43);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

	HnswlibIndexGpu index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
											 kEfConstruction);
	std::vector<std::byte> bytes(vectors.front().size() * vectors.size());
	std::vector<VectorId> ids(vectors.size());
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		std::memcpy(bytes.data() + i * vectors.front().size(), vectors[i].data(), vectors.front().size());
		ids[i] = static_cast<VectorId>(i) + 1;
	}
	index.build(VectorBatchView{bytes.data(), kDim, VectorDType::Float32, ids.size(), ids.data()});

	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/64, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/100, DistanceMetric::L2, VectorDType::Float32);
	Controller controller(index, routing_cache);  // default (generous) budget -- irrelevant, nothing ever gets asked for
	index.registerAllRegions(controller);
	index.attachController(controller);
	// Deliberately no warm-up search loop -- every Region stays host-only.

	for (RegionId region : index.allRegions()) {
		ASSERT_FALSE(controller.acquireRegion(region).on_device) << "region " << region
																															 << " is resident despite no search ever running -- "
																																	"test setup problem, this test needs zero residency";
	}

	std::size_t compared = 0;
	for (std::size_t i = 0; i < vectors.size(); i += 5) {
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i].data(), kDim, VectorDType::Float32};
		request.query.top_k = 3;

		TraverseResult host_result = index.traverseHost({request}).front();
		TraverseResult device_result = index.traverseDevice({request}).front();
		EXPECT_TRUE(device_result.completed_within_scope);
		ASSERT_EQ(host_result.result.neighbors.size(), device_result.result.neighbors.size());

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
		EXPECT_GT(overlap, 0u) << "query index " << i << ": zero overlap under zero residency";
		++compared;
	}
	ASSERT_GT(compared, 0u);
}

// TraverseResult::touched drives RegionManager's promotion/eviction hotness
// signal (Controller::search()'s on_complete -> recordTraversal(), see
// core/controller.cpp) -- before this session's change, traverseDevice()
// left it empty unconditionally, so a device-served query contributed no
// signal at all regardless of how much of the graph it actually walked.
TEST(HnswlibIndexGpuPartialResidencyTest, TraverseDevicePopulatesTouchedRegions) {
	std::mt19937 rng(47);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

	HnswlibIndexGpu index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
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
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, /*top_k=*/1};
		controller.search(query);
	}
	controller.waitIdle();

	std::vector<RegionId> all_regions = index.allRegions();
	std::unordered_set<RegionId> known_regions(all_regions.begin(), all_regions.end());

	TraverseRequest request;
	request.query.vector = VectorView{vectors[0].data(), kDim, VectorDType::Float32};
	request.query.top_k = 1;
	TraverseResult result = index.traverseDevice({request}).front();

	ASSERT_FALSE(result.touched.regions.empty())
			<< "traverseDevice() reported zero touched regions -- device-served queries would contribute no "
				 "hotness signal to RegionManager at all";
	for (RegionId region : result.touched.regions) {
		EXPECT_TRUE(known_regions.count(region)) << "touched an unregistered region " << region;
	}
}

// Verifies TraverseBatchOnDevice()'s hop-synchronized multi-query batching
// (hnswlib_index_gpu.cpp) is a strict generalization of the old
// one-query-at-a-time loop, not a behavior change: each request's search is
// fully independent of every other request sharing its batch (its own
// candidate_set/top_candidates/visited state, its own resolveEntryPoint()
// call), so batching several requests into one traverseDevice() call must
// produce EXACTLY the same per-request result -- not just overlapping, but
// identical neighbor ids in identical order with identical distances -- as
// calling traverseDevice() once per request. This is the load-bearing
// correctness property of the whole redesign: batching several queries'
// rounds into one combined GPU kernel launch must never change what any one
// of them computes.
TEST(HnswlibIndexGpuBatchTest, BatchMatchesSequentialSingleCalls) {
	std::mt19937 rng(53);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

	// max_batch_size=8 -- matches kBatchCount below, so this test exercises
	// the fast (scratch-backed) path, not the oversized-batch fallback (see
	// the fallback-specific test further down for that).
	HnswlibIndexGpu index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
											 kEfConstruction, /*max_batch_size=*/8);
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

	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, /*top_k=*/1};
		controller.search(query);
	}
	controller.waitIdle();

	std::size_t resident_regions = 0;
	for (RegionId region : index.allRegions()) {
		if (controller.acquireRegion(region).on_device) ++resident_regions;
	}
	ASSERT_GT(resident_regions, 0u) << "no Region got promoted -- test setup itself is broken";

	constexpr std::size_t kBatchCount = 8;
	std::vector<TraverseRequest> batch;
	batch.reserve(kBatchCount);
	for (std::size_t i = 0; i < kBatchCount; ++i) {
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i * 23 % vectors.size()].data(), kDim, VectorDType::Float32};
		request.query.top_k = 3;
		batch.push_back(request);
	}

	std::vector<TraverseResult> batched_results = index.traverseDevice(batch);
	ASSERT_EQ(batched_results.size(), kBatchCount);

	for (std::size_t i = 0; i < kBatchCount; ++i) {
		TraverseResult single_result = index.traverseDevice({batch[i]}).front();
		ASSERT_EQ(batched_results[i].result.neighbors.size(), single_result.result.neighbors.size())
				<< "batch index " << i;
		for (std::size_t j = 0; j < single_result.result.neighbors.size(); ++j) {
			EXPECT_EQ(batched_results[i].result.neighbors[j].id, single_result.result.neighbors[j].id)
					<< "batch index " << i << " neighbor " << j
					<< " -- batching changed which candidate this request found, it shouldn't have";
			EXPECT_NEAR(batched_results[i].result.neighbors[j].distance, single_result.result.neighbors[j].distance, 1e-4f)
					<< "batch index " << i << " neighbor " << j;
		}

		std::unordered_set<RegionId> batched_touched(batched_results[i].touched.regions.begin(),
																									batched_results[i].touched.regions.end());
		std::unordered_set<RegionId> single_touched(single_result.touched.regions.begin(),
																								 single_result.touched.regions.end());
		EXPECT_EQ(batched_touched, single_touched) << "batch index " << i << ": touched-region set differs from the "
																											 "single-request call, batching shouldn't change which "
																											 "candidates a request's own walk visits";
	}
}

// Extends the existing single-query device-vs-host comparison (above) to a
// real multi-query batch: every request in one traverseDevice() call must
// still independently match traverseHost() ground truth, the same
// set-overlap-plus-close-distance standard the parameterized test above
// uses for a single request at a time.
TEST(HnswlibIndexGpuBatchTest, BatchOfMultipleQueriesMatchesTraverseHostGroundTruth) {
	std::mt19937 rng(59);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

	HnswlibIndexGpu index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
											 kEfConstruction, /*max_batch_size=*/16);
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

	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, /*top_k=*/1};
		controller.search(query);
	}
	controller.waitIdle();

	constexpr std::size_t kBatchCount = 12;
	std::vector<TraverseRequest> batch;
	batch.reserve(kBatchCount);
	for (std::size_t i = 0; i < kBatchCount; ++i) {
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i * 17 % vectors.size()].data(), kDim, VectorDType::Float32};
		request.query.top_k = 3;
		batch.push_back(request);
	}

	std::vector<TraverseResult> device_results = index.traverseDevice(batch);
	ASSERT_EQ(device_results.size(), kBatchCount);

	for (std::size_t i = 0; i < kBatchCount; ++i) {
		TraverseResult host_result = index.traverseHost({batch[i]}).front();
		const TraverseResult& device_result = device_results[i];
		ASSERT_EQ(host_result.result.neighbors.size(), device_result.result.neighbors.size()) << "batch index " << i;

		std::size_t overlap = 0;
		for (const auto& host_neighbor : host_result.result.neighbors) {
			for (const auto& device_neighbor : device_result.result.neighbors) {
				if (host_neighbor.id != device_neighbor.id) continue;
				++overlap;
				EXPECT_NEAR(host_neighbor.distance, device_neighbor.distance, 1e-2f) << "batch index " << i << " id "
																																							<< host_neighbor.id;
				break;
			}
		}
		EXPECT_GT(overlap, 0u) << "batch index " << i << ": zero overlap between host and device top-"
													 << host_result.result.neighbors.size();
	}
}

// TraverseBatchOnDevice() falls back to a one-off cudaMalloc for whichever
// scratch-backed buffer a call/round exceeds reserved capacity for (see
// ComputeScratchLayout()'s own comment, hnswlib_index_gpu.cpp) -- this test
// deliberately constructs the adapter with a max_batch_size far smaller than
// the batch it's actually asked to serve, forcing both the query buffer and
// the per-round candidate buffers through that fallback, and checks results
// are still correct (still match traverseHost() ground truth) despite never
// touching the fast scratch-backed path at all.
TEST(HnswlibIndexGpuBatchTest, BatchLargerThanMaxBatchSizeFallsBackCorrectly) {
	std::mt19937 rng(61);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, kNumVectors, rng);

	HnswlibIndexGpu index(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
											 kEfConstruction, /*max_batch_size=*/2);
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

	for (std::size_t i = 0; i < vectors.size(); ++i) {
		Query query{VectorView{vectors[i].data(), kDim, VectorDType::Float32}, /*top_k=*/1};
		controller.search(query);
	}
	controller.waitIdle();

	std::size_t resident_regions = 0;
	for (RegionId region : index.allRegions()) {
		if (controller.acquireRegion(region).on_device) ++resident_regions;
	}
	ASSERT_GT(resident_regions, 0u) << "no Region got promoted -- test setup itself is broken";

	constexpr std::size_t kBatchCount = 7;  // > max_batch_size=2, forces the one-off cudaMalloc fallback
	std::vector<TraverseRequest> batch;
	batch.reserve(kBatchCount);
	for (std::size_t i = 0; i < kBatchCount; ++i) {
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i * 29 % vectors.size()].data(), kDim, VectorDType::Float32};
		request.query.top_k = 3;
		batch.push_back(request);
	}

	std::vector<TraverseResult> device_results = index.traverseDevice(batch);
	ASSERT_EQ(device_results.size(), kBatchCount);

	for (std::size_t i = 0; i < kBatchCount; ++i) {
		TraverseResult host_result = index.traverseHost({batch[i]}).front();
		const TraverseResult& device_result = device_results[i];
		EXPECT_TRUE(device_result.completed_within_scope);
		ASSERT_EQ(host_result.result.neighbors.size(), device_result.result.neighbors.size()) << "batch index " << i;

		std::size_t overlap = 0;
		for (const auto& host_neighbor : host_result.result.neighbors) {
			for (const auto& device_neighbor : device_result.result.neighbors) {
				if (host_neighbor.id != device_neighbor.id) continue;
				++overlap;
				EXPECT_NEAR(host_neighbor.distance, device_neighbor.distance, 1e-2f) << "batch index " << i << " id "
																																							<< host_neighbor.id;
				break;
			}
		}
		EXPECT_GT(overlap, 0u) << "batch index " << i << ": zero overlap under the oversized-batch fallback path";
	}
}

// Pure sizing check, no Controller/GPU execution needed: requiredScratchBytesPerWorker()
// must actually grow with max_batch_size (the ctor parameter this session
// added) -- a regression here would mean a caller configuring a larger
// SchedulingConfig::traverse_batch_size silently keeps getting the old
// batch-of-1 scratch reservation, quietly falling back to the slow
// one-off-cudaMalloc path on every call.
TEST(HnswlibIndexGpuBatchTest, RequiredScratchBytesScalesWithMaxBatchSize) {
	HnswlibIndexGpu small_batch(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
													 kEfConstruction, /*max_batch_size=*/1);
	HnswlibIndexGpu large_batch(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
													 kEfConstruction, /*max_batch_size=*/8);
	EXPECT_GT(large_batch.requiredScratchBytesPerWorker(), small_batch.requiredScratchBytesPerWorker() * 2);
}

INSTANTIATE_TEST_SUITE_P(AllDTypesAndMetrics, HnswlibIndexGpuTest,
		testing::Values(DTypeMetric{VectorDType::Int8, DistanceMetric::L2}, DTypeMetric{VectorDType::Int8, DistanceMetric::InnerProduct},
				DTypeMetric{VectorDType::UInt8, DistanceMetric::L2}, DTypeMetric{VectorDType::UInt8, DistanceMetric::InnerProduct},
				DTypeMetric{VectorDType::Float16, DistanceMetric::L2}, DTypeMetric{VectorDType::Float16, DistanceMetric::InnerProduct},
				DTypeMetric{VectorDType::Float32, DistanceMetric::L2}, DTypeMetric{VectorDType::Float32, DistanceMetric::InnerProduct}),
		DTypeMetricName);

}  // namespace
