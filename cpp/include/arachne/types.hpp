#pragma once

#include <cstdint>
#include <vector>

namespace arachne {

using VectorId = std::uint64_t;
using RegionId = std::uint64_t;

/// Non-owning view over a single vector's raw components. Callers own the
/// backing storage; Arachne never assumes it outlives a single call.
struct VectorView {
	const float* data = nullptr;
	std::uint32_t dim = 0;
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
