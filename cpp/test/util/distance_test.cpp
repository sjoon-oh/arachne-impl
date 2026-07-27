#include "util/distance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

using arachne::util::DotProduct;
using arachne::util::Normalize;
using arachne::util::SquaredL2Distance;

// Deliberately not a multiple of any SIMD lane width, on both sides of one
// (dim=1) and past several lane widths (dim=37) -- exercises the scalar
// tail path after the vectorized loop regardless of which target Highway
// dispatches to at runtime.
std::vector<float> Iota(std::uint32_t dim, float start) {
	std::vector<float> v(dim);
	std::iota(v.begin(), v.end(), start);
	return v;
}

float NaiveSquaredL2(const std::vector<float>& a, const std::vector<float>& b) {
	float total = 0.0f;
	for (std::size_t i = 0; i < a.size(); ++i) {
		float diff = a[i] - b[i];
		total += diff * diff;
	}
	return total;
}

float NaiveDot(const std::vector<float>& a, const std::vector<float>& b) {
	float total = 0.0f;
	for (std::size_t i = 0; i < a.size(); ++i) total += a[i] * b[i];
	return total;
}

class SimdDimTest : public testing::TestWithParam<std::uint32_t> {};

TEST_P(SimdDimTest, SquaredL2DistanceMatchesNaive) {
	std::uint32_t dim = GetParam();
	std::vector<float> a = Iota(dim, 1.0f);
	std::vector<float> b = Iota(dim, 3.5f);

	EXPECT_NEAR(SquaredL2Distance(a.data(), b.data(), dim), NaiveSquaredL2(a, b), 1e-2f);
}

TEST_P(SimdDimTest, DotProductMatchesNaive) {
	std::uint32_t dim = GetParam();
	std::vector<float> a = Iota(dim, 1.0f);
	std::vector<float> b = Iota(dim, -2.0f);

	EXPECT_NEAR(DotProduct(a.data(), b.data(), dim), NaiveDot(a, b), 1e-1f);
}

TEST_P(SimdDimTest, NormalizeProducesUnitVectorInSameDirection) {
	std::uint32_t dim = GetParam();
	std::vector<float> v = Iota(dim, 1.0f);
	std::vector<float> unit(dim);

	Normalize(v.data(), unit.data(), dim);

	EXPECT_NEAR(DotProduct(unit.data(), unit.data(), dim), 1.0f, 1e-3f);

	// Same direction as the input: cosine similarity with the original ~1.
	float dot_with_original = DotProduct(unit.data(), v.data(), dim);
	float original_norm = std::sqrt(DotProduct(v.data(), v.data(), dim));
	EXPECT_NEAR(dot_with_original / original_norm, 1.0f, 1e-3f);
}

TEST_P(SimdDimTest, NormalizeSupportsInPlaceAliasing) {
	std::uint32_t dim = GetParam();
	std::vector<float> v = Iota(dim, 1.0f);
	std::vector<float> expected(dim);
	Normalize(v.data(), expected.data(), dim);

	std::vector<float> in_place = Iota(dim, 1.0f);
	Normalize(in_place.data(), in_place.data(), dim);

	for (std::uint32_t i = 0; i < dim; ++i) {
		EXPECT_NEAR(in_place[i], expected[i], 1e-4f) << "index " << i;
	}
}

INSTANTIATE_TEST_SUITE_P(VariousDims, SimdDimTest,
													testing::Values(1, 2, 3, 4, 7, 8, 16, 17, 37, 128, 129));

TEST(SimdTest, NormalizeOfZeroVectorStaysZero) {
	constexpr std::uint32_t kDim = 5;
	std::vector<float> zero(kDim, 0.0f);
	std::vector<float> out(kDim, 42.0f);

	Normalize(zero.data(), out.data(), kDim);

	for (float x : out) EXPECT_EQ(x, 0.0f);
}

TEST(SimdTest, SquaredL2DistanceOfIdenticalVectorsIsZero) {
	constexpr std::uint32_t kDim = 33;
	std::vector<float> v = Iota(kDim, 1.0f);

	EXPECT_EQ(SquaredL2Distance(v.data(), v.data(), kDim), 0.0f);
}

}  // namespace
