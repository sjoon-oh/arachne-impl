#pragma once

#include <cstdint>

namespace arachne::util {

/// SIMD-accelerated vector math (Google Highway underneath, dispatched at
/// runtime to the best available target -- AVX-512/AVX2/SSE4/NEON/scalar --
/// for the CPU actually running, not just the one this was compiled on).
/// Highway is an impl-only dependency: this header declares plain float*
/// signatures, so nothing that only includes it needs Highway on its
/// include path (see distance.cpp).
///
/// These are generic vector-math building blocks for Arachne's own
/// control-plane code (currently ASRoutingCacheHnsw's Cosine-metric
/// normalization; a candidate for Core's future verification-path
/// distance/margin comparisons and promotion/eviction scoring) -- not
/// index-facing primitives. An underlying index's own traversal/
/// modification math stays behind IndexAdapter/IRegion, untouched by this.

/// Squared Euclidean (L2) distance between `a` and `b`, each `dim` floats.
float SquaredL2Distance(const float* a, const float* b, std::uint32_t dim);

/// Dot product of `a` and `b`, each `dim` floats.
float DotProduct(const float* a, const float* b, std::uint32_t dim);

/// Writes `in` L2-normalized into `out` (both `dim` floats; safe to alias,
/// i.e. `out == in`). A zero vector is left as all zeros rather than
/// producing NaN.
void Normalize(const float* in, float* out, std::uint32_t dim);

}  // namespace arachne::util
