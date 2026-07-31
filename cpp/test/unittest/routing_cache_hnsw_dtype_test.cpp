// Covers the (VectorDType, DistanceMetric) matrix ASRoutingCacheHnsw gained
// on top of hnswlib's new int8/uint8/float16 SIMD spaces (see
// thirdparty/hnswlib.patch) -- routing_cache_hnsw_test.cpp already covers
// the pre-existing Float32 L2/Cosine paths in depth, so this file focuses
// on the new dtypes and the dtype/metric validation ASRoutingCacheHnsw does
// at construction and per-call. Test data for the Float16 cases is produced
// by this file's own local ToHalfBits() encoder rather than reusing
// hnswlib's internal half_utils.h (see that function's doc comment).

#include "core/routing_cache_hnsw.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using arachne::DistanceMetric;
using arachne::ASRoutingCacheHnsw;
using arachne::VectorDType;
using arachne::VectorId;
using arachne::VectorView;

constexpr std::uint32_t kDim = 8;
constexpr float kSelfMaxDistance = 1e-3f;  // distance-to-self is always 0

// Minimal local float->binary16 encoder, only to synthesize test data --
// deliberately not reusing hnswlib's half_utils.h, since hnswlib is meant
// to stay a .cpp-only (impl) dependency of arachne_core, not something
// test/application code reaches into directly.
std::uint16_t ToHalfBits(float f) {
	std::uint32_t bits;
	std::memcpy(&bits, &f, sizeof(bits));
	std::uint32_t sign = (bits >> 16) & 0x8000u;
	std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
	std::uint32_t mant = bits & 0x7FFFFFu;
	if (exp <= 0) return static_cast<std::uint16_t>(sign);
	if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
	return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
}

struct DTypeCase {
	VectorDType dtype;
	DistanceMetric metric;
	const char* name;
};

class ASRoutingCacheHnswDTypeTest : public testing::TestWithParam<DTypeCase> {};

TEST_P(ASRoutingCacheHnswDTypeTest, EnsureThenNearestFindsSelf) {
	const DTypeCase& c = GetParam();
	ASRoutingCacheHnsw cache(kDim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2, /*M=*/16,
												 /*ef_construction=*/200, c.metric, c.dtype);

	// Raw byte buffers, one per candidate dtype -- only the one matching
	// c.dtype is actually used, but keeping them side by side makes clear
	// this is the same 8-element pattern reinterpreted per type.
	std::array<std::int8_t, kDim> as_int8{1, -2, 3, -4, 5, -6, 7, -8};
	std::array<unsigned char, kDim> as_uint8{1, 2, 3, 4, 5, 6, 7, 8};
	std::array<std::uint16_t, kDim> as_half{};
	std::array<float, kDim> as_float{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
	for (std::uint32_t i = 0; i < kDim; i++) as_half[i] = ToHalfBits(as_float[i]);

	VectorView v;
	v.dim = kDim;
	v.dtype = c.dtype;
	switch (c.dtype) {
		case VectorDType::Int8:
			v.data = as_int8.data();
			break;
		case VectorDType::UInt8:
			v.data = as_uint8.data();
			break;
		case VectorDType::Float16:
			v.data = as_half.data();
			break;
		case VectorDType::Float32:
			v.data = as_float.data();
			break;
	}

	VectorId id = cache.ensure(1, v, kSelfMaxDistance);
	EXPECT_EQ(id, 1u);
	EXPECT_EQ(cache.nearest(v), 1u);
}

INSTANTIATE_TEST_SUITE_P(
		AllCombinations, ASRoutingCacheHnswDTypeTest,
		testing::Values(DTypeCase{VectorDType::Int8, DistanceMetric::L2, "Int8L2"},
										 DTypeCase{VectorDType::Int8, DistanceMetric::InnerProduct, "Int8IP"},
										 DTypeCase{VectorDType::UInt8, DistanceMetric::L2, "UInt8L2"},
										 DTypeCase{VectorDType::UInt8, DistanceMetric::InnerProduct, "UInt8IP"},
										 DTypeCase{VectorDType::Float16, DistanceMetric::L2, "Float16L2"},
										 DTypeCase{VectorDType::Float16, DistanceMetric::InnerProduct, "Float16IP"},
										 DTypeCase{VectorDType::Float32, DistanceMetric::L2, "Float32L2"},
										 DTypeCase{VectorDType::Float32, DistanceMetric::InnerProduct, "Float32IP"}),
		[](const testing::TestParamInfo<DTypeCase>& info) { return info.param.name; });

TEST(ASRoutingCacheHnswDTypeTest, CosineRejectsNonFloat32Dtype) {
	EXPECT_THROW(ASRoutingCacheHnsw(kDim, 1024, 0.2, 16, 200, DistanceMetric::Cosine, VectorDType::UInt8),
							 std::invalid_argument);
	EXPECT_THROW(ASRoutingCacheHnsw(kDim, 1024, 0.2, 16, 200, DistanceMetric::Cosine, VectorDType::Int8),
							 std::invalid_argument);
	EXPECT_THROW(ASRoutingCacheHnsw(kDim, 1024, 0.2, 16, 200, DistanceMetric::Cosine, VectorDType::Float16),
							 std::invalid_argument);
	EXPECT_NO_THROW(ASRoutingCacheHnsw(kDim, 1024, 0.2, 16, 200, DistanceMetric::Cosine, VectorDType::Float32));
}

TEST(ASRoutingCacheHnswDTypeTest, MismatchedDtypeThrowsOnEnsureAndNearest) {
	ASRoutingCacheHnsw cache(kDim, 1024, 0.2, 16, 200, DistanceMetric::L2, VectorDType::UInt8);

	std::array<float, kDim> wrong_type_data{1, 2, 3, 4, 5, 6, 7, 8};
	VectorView mismatched{wrong_type_data.data(), kDim, VectorDType::Float32};

	EXPECT_THROW(cache.ensure(1, mismatched, kSelfMaxDistance), std::invalid_argument);
	EXPECT_THROW(cache.nearest(mismatched), std::invalid_argument);
}

// Larger-scale, randomized version of EnsureThenNearestFindsSelf for the
// three new dtypes, at a more realistic dim -- mirrors
// routing_cache_hnsw_test.cpp's BuildsAndCompactsWithManyRandomHighDimensionalVectors
// but across dtypes instead of just float32.
TEST(ASRoutingCacheHnswDTypeTest, ManyRandomVectorsPerNewDtype) {
	constexpr std::uint32_t kLargeDim = 64;
	constexpr std::size_t kNumVectors = 100;
	std::mt19937 rng(7);

	{
		std::uniform_int_distribution<int> dist(0, 255);
		ASRoutingCacheHnsw cache(kLargeDim, 128, 0.2, 16, 200, DistanceMetric::L2, VectorDType::UInt8);
		std::vector<std::vector<unsigned char>> vectors(kNumVectors);
		for (std::size_t i = 0; i < kNumVectors; i++) {
			vectors[i].resize(kLargeDim);
			for (auto& x : vectors[i]) x = static_cast<unsigned char>(dist(rng));
			VectorView v{vectors[i].data(), kLargeDim, VectorDType::UInt8};
			ASSERT_EQ(cache.ensure(static_cast<VectorId>(i) + 1, v, kSelfMaxDistance), i + 1) << "uint8 vector " << i;
		}
		for (std::size_t i = 0; i < kNumVectors; i++) {
			VectorView v{vectors[i].data(), kLargeDim, VectorDType::UInt8};
			EXPECT_EQ(cache.nearest(v), i + 1) << "uint8 vector " << i;
		}
	}
	{
		// L2, not InnerProduct: self-distance is always exactly 0 for L2, so
		// one small fixed kSelfMaxDistance reliably distinguishes "this
		// vector" from others. InnerProduct's self-distance is data-dependent
		// (-||v||^2), so it's covered instead by EnsureThenNearestFindsSelf above.
		std::uniform_int_distribution<int> dist(-128, 127);
		ASRoutingCacheHnsw cache(kLargeDim, 128, 0.2, 16, 200, DistanceMetric::L2, VectorDType::Int8);
		std::vector<std::vector<std::int8_t>> vectors(kNumVectors);
		for (std::size_t i = 0; i < kNumVectors; i++) {
			vectors[i].resize(kLargeDim);
			for (auto& x : vectors[i]) x = static_cast<std::int8_t>(dist(rng));
			VectorView v{vectors[i].data(), kLargeDim, VectorDType::Int8};
			ASSERT_EQ(cache.ensure(static_cast<VectorId>(i) + 1, v, kSelfMaxDistance), i + 1) << "int8 vector " << i;
		}
		for (std::size_t i = 0; i < kNumVectors; i++) {
			VectorView v{vectors[i].data(), kLargeDim, VectorDType::Int8};
			EXPECT_EQ(cache.nearest(v), i + 1) << "int8 vector " << i;
		}
	}
	{
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		ASRoutingCacheHnsw cache(kLargeDim, 128, 0.2, 16, 200, DistanceMetric::L2, VectorDType::Float16);
		std::vector<std::vector<std::uint16_t>> vectors(kNumVectors);
		for (std::size_t i = 0; i < kNumVectors; i++) {
			vectors[i].resize(kLargeDim);
			for (auto& x : vectors[i]) x = ToHalfBits(dist(rng));
			VectorView v{vectors[i].data(), kLargeDim, VectorDType::Float16};
			ASSERT_EQ(cache.ensure(static_cast<VectorId>(i) + 1, v, kSelfMaxDistance), i + 1) << "half vector " << i;
		}
		for (std::size_t i = 0; i < kNumVectors; i++) {
			VectorView v{vectors[i].data(), kLargeDim, VectorDType::Float16};
			EXPECT_EQ(cache.nearest(v), i + 1) << "half vector " << i;
		}
	}
}

}  // namespace
