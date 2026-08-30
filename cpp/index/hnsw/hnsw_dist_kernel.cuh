#pragma once

// Host-callable launcher for the batched distance kernels HnswlibIndexGpu's
// copied search loop (hnswlib_index_gpu.cpp) uses to offload per-hop
// candidate distance computation to the GPU -- see that file's own
// overview for the design (report.md §10 "옵션 1" applied to the *device*
// path, i.e. what report.md §7.3/§9 called Phase 2's shallow-offload
// traverseDevice()).
//
// Covers every (VectorDType, DistanceMetric) combination hnswlib itself
// supports except Cosine (Int8/UInt8/Float16/Float32 x L2/InnerProduct --
// see hnsw_dist_kernel.cu's own overview for why Cosine is out of scope
// here, same reason HnswlibIndex::makeEngine() rejects it on the host side).
// Each combination's math mirrors the corresponding hnswlib scalar
// distance function bit-for-bit (thirdparty/hnswlib/hnswlib/space_l2.h /
// space_ip.h's L2Sqr/L2SqrI/L2SqrInt8/L2SqrHalf and
// InnerProduct/InnerProductU8/InnerProductInt8/InnerProductHalf's scalar
// fallback paths, not their SIMD-accelerated variants -- this kernel is
// its own independent implementation of the same formulas, not a call
// into hnswlib) -- see the .cu file for exactly which formula backs each
// case.
//
// Split into its own .cu/.cuh pair (rather than living directly in
// hnswlib_index_gpu.cpp) because it's the one piece here that's actual
// device code (__global__), which plain .cpp translation units can't
// contain.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace arachne::index::hnsw {

// Deliberately not arachne::VectorDType/arachne::DistanceMetric (types.hpp /
// core/routing_cache.hpp): this header is compiled by nvcc, and pulling in
// either of those (transitively: <atomic> and friends) hits an nvcc/glibc
// header incompatibility in this environment (glibc's newer _Float32/
// _Float64/... extended-float typedefs, which some installed nvcc versions'
// preprocessing don't handle). A small self-contained mirror avoids the
// problem entirely; HnswlibIndexGpu::TraverseOneOnDevice() (hnswlib_index_gpu.cpp,
// a plain .cpp with no such restriction) maps its own dtype()/metric() to
// these one-to-one before calling LaunchDistanceKernel().
enum class KernelElemType { Float32, Float16, UInt8, Int8 };
enum class KernelDistanceOp { L2, InnerProduct };

/// Computes the distance (hnswlib's convention: lower is always closer,
/// regardless of metric -- InnerProduct is stored as `1 - dot` (float
/// dtypes) or `-dot` (integer dtypes), matching hnswlib's own
/// InnerProductDistance*() functions exactly, not a raw dot product)
/// between each of `count` candidates and *its own* query vector -- one
/// combined kernel launch batched across every query in a hop-synchronized
/// round of HnswlibIndexGpu's multi-query search (hnswlib_index_gpu.cpp's
/// TraverseBatchOnDevice()), not just one query's candidates the way a
/// single-query walk's round would be. `queries` is `batch_size`
/// consecutive query vectors, `dim` elements of `dtype` each, back-to-back,
/// already on-device; `candidate_query_index` is a device array of `count`
/// batch-local indices (0-based, < `batch_size`) saying which of those
/// queries candidate i's distance is against -- so a single-query call is
/// just the `batch_size == 1` case, every entry of `candidate_query_index`
/// zero. `candidate_ptrs` is a device array of `count` per-vector device
/// pointers (see hnswlib_index_gpu.cpp for how that's built, one Region
/// Lease per distinct Region the round's candidates touch). `queries` and
/// every pointer `candidate_ptrs` contains are typed as `dtype`
/// (VectorElementSize(dtype) bytes per element, types.hpp) -- this
/// function reinterprets them internally, callers pass plain `const
/// void*`/`const void* const*` so this single entry point covers every
/// supported dtype without the caller needing its own dtype switch.
///
/// Writes `count` floats to `out` (on-device) -- always float regardless
/// of dtype (hnswlib's own UInt8/Int8 spaces compute in `int`, matched
/// exactly internally, but the final value handed back is cast to float
/// here so HnswlibIndexGpu's host-side search loop, which already assumes
/// float distances throughout, needs no changes for the integer dtypes).
///
/// One CUDA block per candidate, block-wide parallel reduction over
/// `dim` -- see the .cu file for the kernels themselves; this is
/// deliberately the simplest correct shape (not the fastest), matching
/// this class's own "simplest first, measure before optimizing" scope
/// (report.md §5's SVFusion lesson). Batching multiple queries' candidates
/// into one launch this way changes nothing about that per-candidate
/// shape -- it only means one grid now covers several queries' work
/// instead of one, amortizing the launch/host<->device round-trip itself
/// across the whole batch rather than paying it per query per hop.
///
/// Asynchronous: returns once the kernel is enqueued on `stream`, same
/// convention as the rest of Arachne's GPU code (see
/// gpu/device_region_pool.hpp). Caller is responsible for every pointer's
/// lifetime (queries, candidate_query_index, candidate_ptrs, each
/// candidate itself, out) lasting until the launch completes -- e.g. by
/// holding the relevant gpu::DeviceRegionPool::Lease(s) and
/// syncing/copying `out` back before releasing them. Throws
/// std::invalid_argument for DistanceMetric::Cosine (not a real hnswlib
/// Space -- see this file's own overview).
void LaunchDistanceKernel(KernelElemType dtype, KernelDistanceOp metric, const void* queries,
													 const std::uint32_t* candidate_query_index, const void* const* candidate_ptrs,
													 std::size_t count, std::uint32_t dim, float* out, cudaStream_t stream);

}  // namespace arachne::index::hnsw
