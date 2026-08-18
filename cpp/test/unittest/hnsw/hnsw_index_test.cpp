// Correctness test for HnswIndex (index/hnsw/hnsw_index.hpp/.cpp): build(),
// traverseHost() (search), modifyHost() (insert/delete), and exportTo()/
// loadFrom(), driven directly through the public IAdapter surface (bypassing
// Controller, matching stress_index_export_load_test.cpp's own convention --
// this is about HnswIndex's own contract, not Controller's orchestration).

#include "hnsw_index.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hnsw_index_anchor_entry.hpp"
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
using arachne::index::hnsw::HnswIndex;
using arachne::index::hnsw::HnswIndexAnchorEntry;
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

TraverseResult SearchOne(HnswIndex& index, const void* query_data, std::uint32_t top_k) {
	TraverseRequest request;
	request.query.vector = VectorView{query_data, kDim, index.dtype()};
	request.query.top_k = top_k;
	return index.traverseHost({request}).front();
}

class HnswIndexDTypeTest : public testing::TestWithParam<VectorDType> {};

TEST_P(HnswIndexDTypeTest, BuildThenSearchSelfRecall) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(7);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 200, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswIndex index(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
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

TEST_P(HnswIndexDTypeTest, InsertThenSearchFindsNewVector) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(11);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 50, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswIndex index(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
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

TEST_P(HnswIndexDTypeTest, DeleteThenSearchOmitsDeletedVector) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(13);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 50, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswIndex index(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
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

TEST_P(HnswIndexDTypeTest, ExportLoadRoundTripPreservesSearchResults) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(17);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 150, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswIndex original(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	original.build(dataset.view());

	std::vector<VectorId> before;
	for (std::size_t i = 0; i < 20; ++i) {
		before.push_back(SearchOne(original, vectors[i].data(), 3).result.neighbors.front().id);
	}

	ScratchFile file;
	original.exportTo(file.path());

	HnswIndex reloaded(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	reloaded.loadFrom(file.path());
	EXPECT_EQ(reloaded.liveCount(), original.liveCount());
	EXPECT_EQ(reloaded.allRegions().size(), original.allRegions().size());

	for (std::size_t i = 0; i < 20; ++i) {
		VectorId after = SearchOne(reloaded, vectors[i].data(), 3).result.neighbors.front().id;
		EXPECT_EQ(after, before[i]);
	}
}

TEST_P(HnswIndexDTypeTest, LoadRejectsCapacityMismatch) {
	VectorDType dtype = GetParam();
	std::mt19937 rng(19);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(dtype, kDim, 30, rng);
	FlatDataset dataset = Flatten(vectors, kDim, dtype);

	HnswIndex original(kDim, dtype, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	original.build(dataset.view());
	ScratchFile file;
	original.exportTo(file.path());

	HnswIndex mismatched(kDim, dtype, DistanceMetric::L2, kCapacity * 2, kVectorsPerRegion, kM, kEfConstruction);
	EXPECT_THROW(mismatched.loadFrom(file.path()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllDTypes, HnswIndexDTypeTest,
		testing::Values(VectorDType::Int8, VectorDType::UInt8, VectorDType::Float16, VectorDType::Float32));

// HnswIndexAnchorEntry's entry-point cache and beam width only affect
// traverseDevice() -- traverseHost() is inherited from HnswIndex unmodified
// (hnswlib's own searchKnnCloserFirst(), which has no seam to consult
// resolveEntryPoint()), so its search results must stay identical to plain
// HnswIndex's regardless of whether requests carry an anchor_id. A cheap
// correctness/regression net for that invariant.
TEST(HnswIndexAnchorEntryTest, HostSearchMatchesPlainHnswIndexRegardlessOfAnchorId) {
	std::mt19937 rng(23);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(VectorDType::Float32, kDim, 100, rng);
	FlatDataset dataset = Flatten(vectors, kDim, VectorDType::Float32);

	HnswIndex plain(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM, kEfConstruction);
	plain.build(dataset.view());
	HnswIndexAnchorEntry anchor(kDim, VectorDType::Float32, DistanceMetric::L2, kCapacity, kVectorsPerRegion, kM,
															kEfConstruction);
	anchor.build(dataset.view());

	for (std::size_t i = 0; i < 20; ++i) {
		TraverseRequest request;
		request.query.vector = VectorView{vectors[i].data(), kDim, VectorDType::Float32};
		request.query.top_k = 3;
		request.anchor_id = static_cast<VectorId>(i) + 1;  // exercises the cache-population side effect too
		auto plain_result = plain.traverseHost({request}).front();
		auto anchor_result = anchor.traverseHost({request}).front();
		ASSERT_EQ(plain_result.result.neighbors.size(), anchor_result.result.neighbors.size());
		for (std::size_t k = 0; k < plain_result.result.neighbors.size(); ++k) {
			EXPECT_EQ(plain_result.result.neighbors[k].id, anchor_result.result.neighbors[k].id);
		}
	}
}

}  // namespace
