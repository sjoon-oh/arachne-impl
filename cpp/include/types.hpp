#pragma once

#include <cstdint>
#include <vector>

namespace arachne {

using VectorId = std::uint64_t;
using RegionId = std::uint64_t;

/// The element type a VectorView's bytes are laid out as. Now that
/// hnswlib (thirdparty/hnswlib, see hnswlib.patch) has SIMD-optimized
/// int8/uint8/float16 distance spaces alongside its original float32 one,
/// Arachne threads that choice through as a runtime option rather than
/// assuming float32 everywhere -- ASRoutingCacheHnsw picks the matching
/// hnswlib Space from this (see routing_cache_hnsw.hpp/.cpp).
enum class VectorDType { Int8, UInt8, Float16, Float32 };

/// Bytes one element of `dtype` occupies. Float16 is stored as a raw
/// uint16_t bit pattern (see hnswlib's half_utils.h) -- there is no
/// compiler-native float16 arithmetic type involved.
constexpr std::size_t VectorElementSize(VectorDType dtype) {
	switch (dtype) {
		case VectorDType::Int8:
		case VectorDType::UInt8:
			return 1;
		case VectorDType::Float16:
			return 2;
		case VectorDType::Float32:
			return 4;
	}
	return 0;
}

/// Non-owning view over a single vector's raw components. Callers own the
/// backing storage; Arachne never assumes it outlives a single call. `data`
/// is type-erased -- its actual layout (dim elements of `dtype`, back to
/// back) is described by `dtype`/`dim` rather than the pointer type, so one
/// VectorView shape covers every element type Arachne supports. Defaults
/// to Float32 so existing call sites building a VectorView from a
/// `const float*` don't need to change.
struct VectorView {
	const void* data = nullptr;
	std::uint32_t dim = 0;
	VectorDType dtype = VectorDType::Float32;
};

struct Query {
	VectorView vector;
	std::uint32_t top_k = 0;
};

struct Record {
	VectorId id = 0;
	VectorView vector;
};

struct Neighbor {
	VectorId id = 0;
	float distance = 0.0f;
};

struct SearchResult {
	std::vector<Neighbor> neighbors;
	bool served_gpu_only = false;
};

struct InsertResult {
	bool ok = false;
};

struct DeleteResult {
	bool ok = false;
};

}  // namespace arachne
