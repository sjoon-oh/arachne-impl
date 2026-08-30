#include "hnsw_dist_kernel.cuh"

#include <cstdint>

// See hnsw_dist_kernel.cuh for the design overview. Kernel shape: one CUDA
// block per candidate, kBlockThreads threads per block each summing a
// strided subset of the dim elements, then a standard shared-memory tree
// reduction down to one distance value per block.
//
// Two kernel families, matching which accumulator type hnswlib itself uses
// for each dtype (thirdparty/hnswlib/hnswlib/space_l2.h / space_ip.h):
//   - FloatAccumDistanceKernel<ElemT, IsInnerProduct>: Float32 (ElemT=float,
//     read directly) and Float16 (ElemT=uint16_t, the raw binary16 bit
//     pattern -- decoded via HalfToFloatDevice() below, a device-side port
//     of hnswlib's own HalfToFloat(), half_utils.h) both accumulate in
//     float, matching L2Sqr/L2SqrHalf and InnerProduct/InnerProductHalf's
//     scalar fallback paths exactly.
//   - IntAccumDistanceKernel<ElemT, IsInnerProduct>: UInt8 (ElemT=unsigned
//     char) and Int8 (ElemT=int8_t) both accumulate in int, matching
//     L2SqrI/L2SqrInt8 and InnerProductU8/InnerProductInt8's scalar
//     fallback paths exactly (hnswlib widens each byte to int before any
//     arithmetic for precisely the same overflow reason noted in
//     space_l2.h's own comment on L2SqrU8SIMD16ExtSSE41).
//
// Both families now take `query_index` alongside `candidate_ptrs`: block
// blockIdx.x's candidate looks up its own query via
// `queries + query_index[blockIdx.x] * dim`, instead of every block sharing
// one fixed `query` pointer. This is the entire multi-query batching change
// at the kernel level -- one more strided load per block, nothing else
// about the per-candidate reduction shape below differs from the
// single-query version it replaced (see hnsw_dist_kernel.cuh's own comment
// for why: candidate_query_index is all-zero for a batch_size==1 call, so
// this is a strict generalization, not a new code path for the old case).
// InnerProduct's epilogue differs by accumulator family, matching hnswlib
// bit-for-bit: float dtypes store `1.0f - dot` (InnerProduct/
// InnerProductHalf in space_ip.h), integer dtypes store plain `-dot`
// (InnerProductU8/InnerProductInt8's InnerProductDistanceU8/Int8 -- no
// `1 -` offset). Both conventions share the property hnswlib's own
// heap-based search algorithms (and this file's copied one, hnswlib_index_gpu.cpp)
// rely on: lower value always means closer, for every metric.
//
// Cosine is not covered -- hnswlib has no native Cosine Space (see
// hnswlib_index.cpp's makeEngine() TODO); LaunchDistanceKernel() below throws
// rather than silently mishandling it, same convention as makeEngine().

namespace arachne::index::hnsw {

namespace {

constexpr int kBlockThreads = 128;

// Device-side port of hnswlib's HalfToFloat() (thirdparty/hnswlib/hnswlib/
// half_utils.h) -- same bit manipulation, __uint_as_float() in place of
// half_utils.h's std::memcpy (device code has no host memcpy semantics to
// rely on, but the bit-reinterpretation this performs is identical).
__device__ inline float HalfToFloatDevice(std::uint16_t h) {
	std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
	std::uint32_t exp = (h >> 10) & 0x1Fu;
	std::uint32_t mant = h & 0x3FFu;
	std::uint32_t bits;

	if (exp == 0) {
		if (mant == 0) {
			bits = sign;  // +/-0
		} else {
			exp = 1;
			while ((mant & 0x400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3FFu;
			bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
		}
	} else if (exp == 0x1Fu) {
		bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN
	} else {
		bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
	}
	return __uint_as_float(bits);
}

template <typename ElemT>
__device__ inline float LoadAsFloat(const ElemT* p, std::uint32_t i);

template <>
__device__ inline float LoadAsFloat<float>(const float* p, std::uint32_t i) {
	return p[i];
}

template <>
__device__ inline float LoadAsFloat<std::uint16_t>(const std::uint16_t* p, std::uint32_t i) {
	return HalfToFloatDevice(p[i]);
}

template <typename ElemT>
__device__ inline int LoadAsInt(const ElemT* p, std::uint32_t i) {
	return static_cast<int>(p[i]);
}

// Float32/Float16: accumulate in float, matching L2Sqr/L2SqrHalf (L2) and
// InnerProduct/InnerProductHalf's scalar fallback (IP) exactly.
template <typename ElemT, bool IsInnerProduct>
__global__ void FloatAccumDistanceKernel(const ElemT* queries, const std::uint32_t* query_index,
																					const ElemT* const* candidate_ptrs, std::size_t count, std::uint32_t dim,
																					float* out) {
	std::size_t candidate_index = blockIdx.x;
	if (candidate_index >= count) return;
	const ElemT* candidate = candidate_ptrs[candidate_index];
	const ElemT* query = queries + static_cast<std::size_t>(query_index[candidate_index]) * dim;

	float partial = 0.0f;
	for (std::uint32_t d = threadIdx.x; d < dim; d += kBlockThreads) {
		float q = LoadAsFloat<ElemT>(query, d);
		float c = LoadAsFloat<ElemT>(candidate, d);
		if (IsInnerProduct) {
			partial += q * c;
		} else {
			float diff = q - c;
			partial += diff * diff;
		}
	}

	__shared__ float shared[kBlockThreads];
	shared[threadIdx.x] = partial;
	__syncthreads();
	for (int stride = kBlockThreads / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		float total = shared[0];
		out[candidate_index] = IsInnerProduct ? (1.0f - total) : total;
	}
}

// UInt8/Int8: accumulate in int, matching L2SqrI/L2SqrInt8 (L2) and
// InnerProductU8/InnerProductInt8's scalar fallback (IP) exactly --
// including IP's epilogue, which is plain `-dot` here (no `1 -` offset,
// unlike the float family above).
template <typename ElemT, bool IsInnerProduct>
__global__ void IntAccumDistanceKernel(const ElemT* queries, const std::uint32_t* query_index,
																				const ElemT* const* candidate_ptrs, std::size_t count, std::uint32_t dim,
																				float* out) {
	std::size_t candidate_index = blockIdx.x;
	if (candidate_index >= count) return;
	const ElemT* candidate = candidate_ptrs[candidate_index];
	const ElemT* query = queries + static_cast<std::size_t>(query_index[candidate_index]) * dim;

	int partial = 0;
	for (std::uint32_t d = threadIdx.x; d < dim; d += kBlockThreads) {
		int q = LoadAsInt<ElemT>(query, d);
		int c = LoadAsInt<ElemT>(candidate, d);
		if (IsInnerProduct) {
			partial += q * c;
		} else {
			int diff = q - c;
			partial += diff * diff;
		}
	}

	__shared__ int shared[kBlockThreads];
	shared[threadIdx.x] = partial;
	__syncthreads();
	for (int stride = kBlockThreads / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		int total = shared[0];
		out[candidate_index] = IsInnerProduct ? static_cast<float>(-total) : static_cast<float>(total);
	}
}

}  // namespace

void LaunchDistanceKernel(KernelElemType dtype, KernelDistanceOp metric, const void* queries,
													 const std::uint32_t* candidate_query_index, const void* const* candidate_ptrs,
													 std::size_t count, std::uint32_t dim, float* out, cudaStream_t stream) {
	if (count == 0) return;
	bool is_inner_product = (metric == KernelDistanceOp::InnerProduct);
	dim3 grid(static_cast<unsigned int>(count));
	dim3 block(kBlockThreads);

	switch (dtype) {
		case KernelElemType::Float32: {
			const auto* q = static_cast<const float*>(queries);
			const auto* const* c = reinterpret_cast<const float* const*>(candidate_ptrs);
			if (is_inner_product) {
				FloatAccumDistanceKernel<float, true><<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			} else {
				FloatAccumDistanceKernel<float, false><<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			}
			break;
		}
		case KernelElemType::Float16: {
			const auto* q = static_cast<const std::uint16_t*>(queries);
			const auto* const* c = reinterpret_cast<const std::uint16_t* const*>(candidate_ptrs);
			if (is_inner_product) {
				FloatAccumDistanceKernel<std::uint16_t, true>
						<<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			} else {
				FloatAccumDistanceKernel<std::uint16_t, false>
						<<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			}
			break;
		}
		case KernelElemType::UInt8: {
			const auto* q = static_cast<const unsigned char*>(queries);
			const auto* const* c = reinterpret_cast<const unsigned char* const*>(candidate_ptrs);
			if (is_inner_product) {
				IntAccumDistanceKernel<unsigned char, true>
						<<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			} else {
				IntAccumDistanceKernel<unsigned char, false>
						<<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			}
			break;
		}
		case KernelElemType::Int8: {
			const auto* q = static_cast<const std::int8_t*>(queries);
			const auto* const* c = reinterpret_cast<const std::int8_t* const*>(candidate_ptrs);
			if (is_inner_product) {
				IntAccumDistanceKernel<std::int8_t, true>
						<<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			} else {
				IntAccumDistanceKernel<std::int8_t, false>
						<<<grid, block, 0, stream>>>(q, candidate_query_index, c, count, dim, out);
			}
			break;
		}
	}
}

}  // namespace arachne::index::hnsw
