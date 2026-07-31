#pragma once

// Shared helpers across the StressIndex test stages (stress_index_test.cpp,
// stress_index_stage2_test.cpp, ...) -- not part of StressIndex itself,
// since generating dtype-encoded test data is a test concern, not something
// the adapter needs.

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "types.hpp"

namespace arachne::stress::testsupport {

/// Generates `count` random, dtype-encoded vectors (each `dim` elements),
/// returned as raw byte buffers the caller keeps alive -- StressIndex/
/// RoutingCache only ever borrow a pointer into these, exactly like a real
/// caller's vector storage would be borrowed (HostRegionView: Arachne never
/// owns host memory).
inline std::vector<std::vector<std::byte>> GenerateVectors(VectorDType dtype, std::uint32_t dim, std::size_t count,
																														std::mt19937& rng) {
	std::vector<std::vector<std::byte>> vectors(count);
	std::size_t elem_size = VectorElementSize(dtype);
	for (auto& v : vectors) v.resize(dim * elem_size);

	switch (dtype) {
		case VectorDType::Int8: {
			std::uniform_int_distribution<int> dist(-128, 127);
			for (auto& v : vectors)
				for (std::uint32_t d = 0; d < dim; ++d) reinterpret_cast<std::int8_t*>(v.data())[d] = static_cast<std::int8_t>(dist(rng));
			break;
		}
		case VectorDType::UInt8: {
			std::uniform_int_distribution<int> dist(0, 255);
			for (auto& v : vectors)
				for (std::uint32_t d = 0; d < dim; ++d) reinterpret_cast<std::uint8_t*>(v.data())[d] = static_cast<std::uint8_t>(dist(rng));
			break;
		}
		case VectorDType::Float16: {
			// Minimal local float->binary16 encoder, only to synthesize test
			// data -- deliberately not reusing hnswlib's half_utils.h, mirroring
			// routing_cache_hnsw_dtype_test.cpp's own ToHalfBits() (hnswlib stays
			// a .cpp-only dependency of arachne_core, not something test code
			// reaches into directly).
			std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
			auto to_half = [](float f) -> std::uint16_t {
				std::uint32_t bits;
				std::memcpy(&bits, &f, sizeof(bits));
				std::uint32_t sign = (bits >> 16) & 0x8000u;
				std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
				std::uint32_t mant = bits & 0x7FFFFFu;
				if (exp <= 0) return static_cast<std::uint16_t>(sign);
				if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
				return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
			};
			for (auto& v : vectors)
				for (std::uint32_t d = 0; d < dim; ++d) reinterpret_cast<std::uint16_t*>(v.data())[d] = to_half(dist(rng));
			break;
		}
		case VectorDType::Float32: {
			std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
			for (auto& v : vectors)
				for (std::uint32_t d = 0; d < dim; ++d) reinterpret_cast<float*>(v.data())[d] = dist(rng);
			break;
		}
	}
	return vectors;
}

}  // namespace arachne::stress::testsupport
