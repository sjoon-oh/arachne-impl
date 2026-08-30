// Correctness test for HnswlibIndex (index/hnsw/hnswlib_index.hpp/.cpp): build(),
// traverseHost() (search), modifyHost() (insert/delete), and exportTo()/
// loadFrom(), driven directly through the public IAdapter surface (bypassing
// Controller, matching stress_index_export_load_test.cpp's own convention --
// this is about HnswlibIndex's own contract, not Controller's orchestration).

#include "hnswlib_index.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "stress_test_support.hpp"

namespace {

using arachne::DistanceMetric;
using arachne::ModifyOp;
using arachne::ModifyRequest;
using arachne::Query;
using arachne::Record;
using arachne::TraverseRequest;
using arachne::TraverseResult;
using arachne::VectorBatchView;
using arachne::VectorDType;
using arachne::VectorId;
using arachne::VectorView;
using arachne::index::hnsw::HnswlibIndex;
using arachne::stress::testsupport::GenerateVectors;

constexpr std::uint32_t kDim = 16;
constexpr std::size_t kCapacity = 500;
constexpr std::size_t kVectorsPerRegion = 32;
constexpr std::size_t kM = 16;
constexpr std::size_t kEfConstruction = 100;

class ScratchFile {
 public:
	ScratchFile() {
		const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
		std::string name = std::string(info->test_suite_name()) + "_" + info->name();
		for (char& c : name) {
			if (c == '/') c = '_';
		}
		path_ = testing::TempDir() + "arachne_hnsw_" + name + ".bin";
	}
	~ScratchFile() { std::remove(path_.c_str()); }
	ScratchFile(const ScratchFile&) = delete;
	ScratchFile& operator=(const ScratchFile&) = delete;

	const std::string& path() const { return path_; }

 private:
	std::string path_;
};

// Flattens `vectors` (one std::vector<std::byte> per vector, as
// GenerateVectors() returns) into one contiguous buffer + explicit ids
// 1..count, matching VectorBatchView's contract.
struct FlatDataset {
	std::vector<std::byte> bytes;
	std::vector<VectorId> ids;
	std::uint32_t dim = 0;
	VectorDType dtype = VectorDType::Float32;

	VectorBatchView view() const {
		return VectorBatchView{bytes.data(), dim, dtype, ids.size(), ids.data()};
	}
};

FlatDataset Flatten(const std::vector<std::vector<std::byte>>& vectors, std::uint32_t dim, VectorDType dtype) {
	FlatDataset dataset;
	dataset.dim = dim;
	dataset.dtype = dtype;
	if (vectors.empty()) return dataset;
	std::size_t stride = vectors.front().size();
	dataset.bytes.resize(stride * vectors.size());
	dataset.ids.resize(vectors.size());
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		std::memcpy(dataset.bytes.data() + i * stride, vectors[i].data(), stride);
		dataset.ids[i] = static_cast<VectorId>(i) + 1;
	}
	return dataset;
}

TraverseResult SearchOne(HnswlibIndex& index, const void* query_data, std::uint32_t top_k) {
	TraverseRequest request;
	request.query.vector = VectorView{query_data, kDim, index.dtype()};
	request.query.top_k = top_k;
	return index.traverseHost({request}).front();
}

class HnswlibIndexDTypeTest : public testing::TestWithParam<VectorDType> {};

TEST_P(HnswlibIndexDTypeTest, BuildThenSearchSelfRecall) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(7);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 200, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswlibIndex index(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	index.build(dataset.view());
	EXPECT_EQ(index.liveCount(), vectors.size());

	std::size_t misses = 0;
	for (std::size_t i = 0; i < vectors.size(); ++i) {
		TraverseResult result = SearchOne(index, vectors[i].data(), 1);
		ASSERT_FALSE(result.result.neighbors.empty());
		if (result.result.neighbors.front().id != dataset.ids[i]) ++misses;
	}
	// Self-recall: querying with a point's own vector should almost always
	// return itself as the single nearest neighbor -- matches the tolerance
	// thirdparty/hnswlib/tests/cpp/dtype_space_test.cpp's own graph
	// self-recall checks use (<=10/200 misses there).
	EXPECT_LE(misses, 10u) << "self-recall misses: " << misses << "/" << vectors.size();
}

TEST_P(HnswlibIndexDTypeTest, InsertThenSearchFindsNewVector) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(11);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 50, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswlibIndex index(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	index.build(dataset.view());

	std::vector<std::vector<std::byte>> extra = GenerateVectors(dtype, kDim, 1, rng);
	ModifyRequest insert_request;
	insert_request.op = ModifyOp::Insert;
	insert_request.record.id = 999;
	insert_request.record.vector = VectorView{extra.front().data(), kDim, dtype};
	auto modify_results = index.modifyHost({insert_request});
	ASSERT_TRUE(modify_results.front().ok);
	EXPECT_EQ(index.liveCount(), vectors.size() + 1);

	TraverseResult result = SearchOne(index, extra.front().data(), 1);
	ASSERT_FALSE(result.result.neighbors.empty());
	EXPECT_EQ(result.result.neighbors.front().id, 999u);
}

TEST_P(HnswlibIndexDTypeTest, DeleteThenSearchOmitsDeletedVector) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(13);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 50, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswlibIndex index(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	index.build(dataset.view());

	ModifyRequest delete_request;
	delete_request.op = ModifyOp::Delete;
	delete_request.target = dataset.ids.front();  // id 1
	auto modify_results = index.modifyHost({delete_request});
	ASSERT_TRUE(modify_results.front().ok);
	EXPECT_EQ(index.liveCount(), vectors.size() - 1);

	// Querying the deleted vector's own data should no longer surface its id.
	TraverseResult result = SearchOne(index, vectors.front().data(), 5);
	for (const auto& neighbor : result.result.neighbors) {
		EXPECT_NE(neighbor.id, dataset.ids.front());
	}
}

TEST_P(HnswlibIndexDTypeTest, ExportLoadRoundTripPreservesSearchResults) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(17);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 150, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswlibIndex original(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	original.build(dataset.view());

	std::vector<VectorId> before;
	for (std::size_t i = 0; i < 20; ++i) {
		before.push_back(SearchOne(original, vectors[i].data(), 3).result.neighbors.front().id);
	}

	ScratchFile file;
	original.exportTo(file.path());

	HnswlibIndex reloaded(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	reloaded.loadFrom(file.path());
	EXPECT_EQ(reloaded.liveCount(), original.liveCount());
	EXPECT_EQ(reloaded.allRegions().size(), original.allRegions().size());

	for (std::size_t i = 0; i < 20; ++i) {
		VectorId after = SearchOne(reloaded, vectors[i].data(), 3).result.neighbors.front().id;
		EXPECT_EQ(after, before[i]);
	}
}

TEST_P(HnswlibIndexDTypeTest, LoadRejectsCapacityMismatch) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(19);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 30, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswlibIndex original(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	original.build(dataset.view());
	ScratchFile file;
	original.exportTo(file.path());

	HnswlibIndex mismatched(kDim, dtype, DistanceMetric::L2, kCapacity * 2, kVectorsPerRegion, kM, kEfConstruction);
	EXPECT_THROW(mismatched.loadFrom(file.path()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllDTypes, HnswlibIndexDTypeTest,
		testing::Values(VectorDType::Int8, VectorDType::UInt8, VectorDType::Float16, VectorDType::Float32));

}  // namespace
