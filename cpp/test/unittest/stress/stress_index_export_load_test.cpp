// Verifies StressIndex::exportTo()/loadFrom() (the concrete implementation of
// IAdapter::exportTo()/loadFrom(), see that interface's own doc comment for
// the general contract) actually round-trips a StressIndex's full state --
// buffer_, deleted_, id_to_slot_, next_free_slot_ -- byte-for-byte
// equivalent in observable behavior, not just "doesn't crash". Also covers
// the failure modes a caller is expected to rely on: a shape mismatch
// between the exporting and loading instance must be rejected rather than
// silently truncated/misread, loadFrom() must leave prior state untouched on
// failure, and loadFrom() must fully overwrite rather than merge with
// whatever was already live.

#include "stress_index.hpp"

#include <gtest/gtest.h>

#include <cstdio>
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
using arachne::ModifyOp;
using arachne::ModifyRequest;
using arachne::ModifyResult;
using arachne::Neighbor;
using arachne::Query;
using arachne::Record;
using arachne::SearchResult;
using arachne::TraverseRequest;
using arachne::TraverseResult;
using arachne::VectorDType;
using arachne::VectorId;
using arachne::VectorView;
using arachne::stress::BruteForceGroundTruth;
using arachne::stress::StressIndex;
using arachne::stress::testsupport::GenerateVectors;

constexpr std::uint32_t kDim = 32;

// Unique-per-test-case scratch file, cleaned up on scope exit regardless of
// how the test ends (assertion failure or otherwise unwinds through here).
class ScratchFile {
 public:
	ScratchFile() {
		const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
		std::string name = std::string(info->test_suite_name()) + "_" + info->name();
		for (char& c : name) {
			if (c == '/') c = '_';
		}
		path_ = testing::TempDir() + "arachne_" + name + ".bin";
	}
	~ScratchFile() { std::remove(path_.c_str()); }

	ScratchFile(const ScratchFile&) = delete;
	ScratchFile& operator=(const ScratchFile&) = delete;

	const std::string& path() const { return path_; }

 private:
	std::string path_;
};

// Inserts `count` of `vectors` (ids 1..count) into `index` via the public
// IAdapter surface (modifyHost()), bypassing Controller -- this test is
// about StressIndex's own export/load contract, not Controller's
// orchestration (that's stress_index_test.cpp's job).
void InsertAll(StressIndex& index, const std::vector<std::vector<std::byte>>& vectors, VectorDType dtype) {
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		ModifyRequest request;
		request.op = ModifyOp::Insert;
		request.record.id = static_cast<VectorId>(i) + 1;
		request.record.vector = VectorView{vectors[i].data(), kDim, dtype};
		std::vector<ModifyResult> results = index.modifyHost({request});
		ASSERT_EQ(results.size(), 1u);
		ASSERT_TRUE(results[0].ok) << "insert failed for vector " << i;
	}
}

void DeleteOne(StressIndex& index, VectorId id) {
	ModifyRequest request;
	request.op = ModifyOp::Delete;
	request.target = id;
	std::vector<ModifyResult> results = index.modifyHost({request});
	ASSERT_EQ(results.size(), 1u);
	ASSERT_TRUE(results[0].ok) << "delete failed for id " << id;
}

class StressIndexExportLoadTest : public testing::TestWithParam<VectorDType> {};

TEST_P(StressIndexExportLoadTest, RoundTripPreservesSearchResults) {
	VectorDType dtype = GetParam();
	constexpr std::size_t kCapacity = 500;
	constexpr std::size_t kVectorsPerRegion = 32;
	constexpr std::size_t kNumVectors = 300;
	constexpr std::uint32_t kTopK = 5;

	std::mt19937 rng(7);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, kNumVectors, rng);

	StressIndex source(kDim, dtype, kCapacity, kVectorsPerRegion);
	InsertAll(source, vectors, dtype);
	// Delete a handful so exportTo()/loadFrom() are exercised against a
	// non-trivial deleted_ bitmap and a next_free_slot_ that has gaps behind
	// it (StressIndex never recycles slots -- see its own doc comment).
	for (VectorId id : {VectorId{3}, VectorId{40}, VectorId{101}}) DeleteOne(source, id);
	ASSERT_EQ(source.liveCount(), kNumVectors - 3);

	ScratchFile file;
	source.exportTo(file.path());

	StressIndex loaded(kDim, dtype, kCapacity, kVectorsPerRegion);
	loaded.loadFrom(file.path());

	EXPECT_EQ(loaded.liveCount(), source.liveCount());

	// Every surviving query must find the identical top-k on the loaded copy
	// as an independent brute-force scan of the *original* -- proves
	// buffer_/deleted_/id_to_slot_ all survived the round trip, not just one
	// of them.
	for (std::size_t i = 0; i < kNumVectors; i += 17) {  // sample, not exhaustive (kNumVectors^2 scans add up)
		VectorView query_view{vectors[i].data(), kDim, dtype};
		std::vector<Neighbor> expected = BruteForceGroundTruth(source, query_view, kTopK);

		TraverseRequest request;
		request.query = Query{query_view, kTopK};
		std::vector<TraverseResult> results = loaded.traverseHost({request});
		ASSERT_EQ(results.size(), 1u);
		const std::vector<Neighbor>& actual = results[0].result.neighbors;

		ASSERT_EQ(actual.size(), expected.size()) << "query source vector " << i;
		for (std::size_t k = 0; k < expected.size(); ++k) {
			EXPECT_EQ(actual[k].id, expected[k].id) << "query " << i << " rank " << k;
			EXPECT_FLOAT_EQ(actual[k].distance, expected[k].distance) << "query " << i << " rank " << k;
		}
	}
}

INSTANTIATE_TEST_SUITE_P(AllDTypes, StressIndexExportLoadTest,
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

// End-to-end: a loaded StressIndex must be just as usable by
// Controller/RegionManager as a freshly-built one -- registerAllRegions()
// after loadFrom() and a real Controller::search() must agree with brute
// force, exactly like stress_index_test.cpp's stage-1 case (this is the
// entire point of exportTo()/loadFrom() existing: a caller should be able to
// restart the process and keep serving from disk without Arachne itself
// knowing anything changed).
TEST(StressIndexExportLoadControllerTest, LoadedIndexServesCorrectSearchResultsThroughController) {
	constexpr VectorDType kDtype = VectorDType::Float32;
	constexpr std::size_t kCapacity = 500;
	constexpr std::size_t kVectorsPerRegion = 32;
	constexpr std::size_t kNumVectors = 200;
	constexpr std::uint32_t kTopK = 5;

	std::mt19937 rng(11);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(kDtype, kDim, kNumVectors, rng);

	StressIndex source(kDim, kDtype, kCapacity, kVectorsPerRegion);
	InsertAll(source, vectors, kDtype);

	ScratchFile file;
	source.exportTo(file.path());

	StressIndex loaded(kDim, kDtype, kCapacity, kVectorsPerRegion);
	loaded.loadFrom(file.path());

	ASRoutingCacheHnsw routing_cache(kDim, /*initial_capacity=*/256, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/200, DistanceMetric::L2, kDtype);
	std::size_t element_size = arachne::VectorElementSize(kDtype);
	std::size_t region_bytes = kVectorsPerRegion * kDim * element_size;
	std::size_t num_regions = (kCapacity + kVectorsPerRegion - 1) / kVectorsPerRegion;
	// Comfortably over the whole dataset -- this test is about correctness
	// after load, not eviction behavior.
	std::size_t budget = region_bytes * num_regions * 2;

	Controller controller(loaded, routing_cache, arachne::SchedulingConfig{}, nullptr, budget);
	loaded.registerAllRegions(controller);

	for (std::size_t i = 0; i < kNumVectors; i += 13) {
		VectorView query_view{vectors[i].data(), kDim, kDtype};
		Query query{query_view, kTopK};
		SearchResult searched = controller.search(query);
		std::vector<Neighbor> expected = BruteForceGroundTruth(source, query_view, kTopK);

		ASSERT_EQ(searched.neighbors.size(), expected.size()) << "query source vector " << i;
		for (std::size_t k = 0; k < expected.size(); ++k) {
			EXPECT_EQ(searched.neighbors[k].id, expected[k].id) << "query " << i << " rank " << k;
			EXPECT_FLOAT_EQ(searched.neighbors[k].distance, expected[k].distance) << "query " << i << " rank " << k;
		}
	}
}

TEST(StressIndexExportLoadEdgeCasesTest, EmptyIndexRoundTrips) {
	constexpr VectorDType kDtype = VectorDType::Float32;
	StressIndex source(kDim, kDtype, /*capacity=*/50, /*vectors_per_region=*/10);

	ScratchFile file;
	source.exportTo(file.path());

	StressIndex loaded(kDim, kDtype, /*capacity=*/50, /*vectors_per_region=*/10);
	loaded.loadFrom(file.path());
	EXPECT_EQ(loaded.liveCount(), 0u);
}

TEST(StressIndexExportLoadEdgeCasesTest, LoadOverwritesRatherThanMerges) {
	constexpr VectorDType kDtype = VectorDType::Float32;
	constexpr std::size_t kCapacity = 50;
	constexpr std::size_t kVectorsPerRegion = 10;

	std::mt19937 rng(3);
	std::vector<std::vector<std::byte>> a_vectors = GenerateVectors(kDtype, kDim, 5, rng);
	std::vector<std::vector<std::byte>> b_vectors = GenerateVectors(kDtype, kDim, 5, rng);

	StressIndex a(kDim, kDtype, kCapacity, kVectorsPerRegion);
	InsertAll(a, a_vectors, kDtype);
	ScratchFile file;
	a.exportTo(file.path());

	// `b` already has its own (different id-space-colliding) live data before
	// loadFrom() runs -- if loadFrom() merged instead of overwrote, b would
	// end up with both a's and its own original data live.
	StressIndex b(kDim, kDtype, kCapacity, kVectorsPerRegion);
	InsertAll(b, b_vectors, kDtype);
	ASSERT_EQ(b.liveCount(), 5u);

	b.loadFrom(file.path());
	EXPECT_EQ(b.liveCount(), a.liveCount());

	VectorView query_view{a_vectors[0].data(), kDim, kDtype};
	TraverseRequest request;
	request.query = Query{query_view, /*top_k=*/1};
	std::vector<TraverseResult> results = b.traverseHost({request});
	ASSERT_EQ(results.size(), 1u);
	ASSERT_FALSE(results[0].result.neighbors.empty());
	EXPECT_EQ(results[0].result.neighbors.front().id, VectorId{1});
	EXPECT_FLOAT_EQ(results[0].result.neighbors.front().distance, 0.0f);
}

TEST(StressIndexExportLoadEdgeCasesTest, LoadRejectsDimensionMismatch) {
	StressIndex source(kDim, VectorDType::Float32, /*capacity=*/50, /*vectors_per_region=*/10);
	ScratchFile file;
	source.exportTo(file.path());

	StressIndex mismatched(kDim + 1, VectorDType::Float32, /*capacity=*/50, /*vectors_per_region=*/10);
	EXPECT_THROW(mismatched.loadFrom(file.path()), std::invalid_argument);
}

TEST(StressIndexExportLoadEdgeCasesTest, LoadRejectsCapacityMismatch) {
	StressIndex source(kDim, VectorDType::Float32, /*capacity=*/50, /*vectors_per_region=*/10);
	ScratchFile file;
	source.exportTo(file.path());

	StressIndex mismatched(kDim, VectorDType::Float32, /*capacity=*/64, /*vectors_per_region=*/10);
	EXPECT_THROW(mismatched.loadFrom(file.path()), std::invalid_argument);
}

TEST(StressIndexExportLoadEdgeCasesTest, LoadRejectsMissingFile) {
	StressIndex index(kDim, VectorDType::Float32, /*capacity=*/50, /*vectors_per_region=*/10);
	EXPECT_THROW(index.loadFrom(testing::TempDir() + "arachne_this_file_does_not_exist.bin"), std::runtime_error);
}

TEST(StressIndexExportLoadEdgeCasesTest, LoadLeavesStateUntouchedOnFailure) {
	constexpr VectorDType kDtype = VectorDType::Float32;
	std::mt19937 rng(5);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(kDtype, kDim, 3, rng);

	StressIndex index(kDim, kDtype, /*capacity=*/50, /*vectors_per_region=*/10);
	InsertAll(index, vectors, kDtype);
	ASSERT_EQ(index.liveCount(), 3u);

	EXPECT_THROW(index.loadFrom(testing::TempDir() + "arachne_this_file_does_not_exist.bin"), std::runtime_error);
	// A failed loadFrom() must not have mutated anything.
	EXPECT_EQ(index.liveCount(), 3u);
}

TEST(StressIndexExportLoadEdgeCasesTest, ExportRejectsUnwritablePath) {
	StressIndex index(kDim, VectorDType::Float32, /*capacity=*/50, /*vectors_per_region=*/10);
	EXPECT_THROW(index.exportTo("/nonexistent_directory_arachne_test/out.bin"), std::runtime_error);
}

}  // namespace
