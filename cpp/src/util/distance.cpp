#include "util/distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

// Highway's multi-target machinery: this file is recompiled once per CPU
// target it's told to support (AVX-512/AVX2/SSE4/NEON/scalar), via
// foreach_target.h re-including it (by the path below, resolved through
// arachne_core's own include dirs) with a different HWY_TARGET value each
// pass -- each pass emits its *Impl functions into its own HWY_NAMESPACE
// (e.g. arachne::util::N_AVX2), so all targets coexist in the same
// translation unit without symbol collisions. A final pass, guarded by
// HWY_ONCE below, compiles once (not per-target) to define the public
// dispatch wrappers: HWY_EXPORT() registers each target's *Impl under one
// table, and HWY_DYNAMIC_DISPATCH() picks the best one for the CPU actually
// running the process, the first time each wrapper is called. See
// https://github.com/google/highway/blob/master/g3doc/quick_reference.md.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/util/distance.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace arachne {
namespace util {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

float SquaredL2DistanceImpl(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
														 std::uint32_t dim) {
	const hn::ScalableTag<float> d;
	const std::size_t lanes = hn::Lanes(d);
	const std::size_t n = dim;

	auto acc = hn::Zero(d);
	std::size_t i = 0;
	for (; i + lanes <= n; i += lanes) {
		const auto va = hn::LoadU(d, a + i);
		const auto vb = hn::LoadU(d, b + i);
		const auto diff = hn::Sub(va, vb);
		acc = hn::MulAdd(diff, diff, acc);
	}

	float total = hn::ReduceSum(d, acc);
	for (; i < n; ++i) {
		const float diff = a[i] - b[i];
		total += diff * diff;
	}
	return total;
}

float DotProductImpl(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, std::uint32_t dim) {
	const hn::ScalableTag<float> d;
	const std::size_t lanes = hn::Lanes(d);
	const std::size_t n = dim;

	auto acc = hn::Zero(d);
	std::size_t i = 0;
	for (; i + lanes <= n; i += lanes) {
		const auto va = hn::LoadU(d, a + i);
		const auto vb = hn::LoadU(d, b + i);
		acc = hn::MulAdd(va, vb, acc);
	}

	float total = hn::ReduceSum(d, acc);
	for (; i < n; ++i) {
		total += a[i] * b[i];
	}
	return total;
}

void NormalizeImpl(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, std::uint32_t dim) {
	const float norm_sq = DotProductImpl(in, in, dim);
	if (norm_sq <= 0.0f) {
		std::fill(out, out + dim, 0.0f);
		return;
	}
	const float inv_norm = 1.0f / std::sqrt(norm_sq);

	const hn::ScalableTag<float> d;
	const std::size_t lanes = hn::Lanes(d);
	const std::size_t n = dim;
	const auto scale = hn::Set(d, inv_norm);

	std::size_t i = 0;
	for (; i + lanes <= n; i += lanes) {
		const auto v = hn::LoadU(d, in + i);
		hn::StoreU(hn::Mul(v, scale), d, out + i);
	}
	for (; i < n; ++i) {
		out[i] = in[i] * inv_norm;
	}
}

}  // namespace HWY_NAMESPACE
}  // namespace util
}  // namespace arachne
HWY_AFTER_NAMESPACE();

// Dispatch wrappers (public arachne::util:: entry points declared in
// distance.hpp) -- see the file-level overview above for why this section
// is guarded by HWY_ONCE.
#if HWY_ONCE
namespace arachne {
namespace util {

HWY_EXPORT(SquaredL2DistanceImpl);
HWY_EXPORT(DotProductImpl);
HWY_EXPORT(NormalizeImpl);

float SquaredL2Distance(const float* a, const float* b, std::uint32_t dim) {
	return HWY_DYNAMIC_DISPATCH(SquaredL2DistanceImpl)(a, b, dim);
}

float DotProduct(const float* a, const float* b, std::uint32_t dim) {
	return HWY_DYNAMIC_DISPATCH(DotProductImpl)(a, b, dim);
}

void Normalize(const float* in, float* out, std::uint32_t dim) {
	HWY_DYNAMIC_DISPATCH(NormalizeImpl)(in, out, dim);
}

}  // namespace util
}  // namespace arachne
#endif  // HWY_ONCE
