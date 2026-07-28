#pragma once

#include <cstddef>

#include <cuda/memory_resource>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include "gpu/stitch_handle.hpp"

namespace arachne::gpu {

/// Reserved up front, once, at DeviceContext construction, when `policy` is
/// AllocationPolicy::Pooled -- see dataResource()/metadataResource() and
/// AllocationPolicy below. Each is only the *initial* reservation; the
/// underlying rmm::mr::pool_memory_resource still grows on demand
/// (coalescing more from the upstream cuda_memory_resource) if a pool's
/// allocations exceed it, so these are sizing hints to avoid early
/// cudaMalloc round-trips, not hard caps. Ignored entirely under
/// AllocationPolicy::Naive, which never pre-reserves anything. Placeholder
/// values pending real budget tuning against actual GPU memory (e.g. via
/// cudaMemGetInfo) and workload characteristics.
inline constexpr std::size_t kDefaultDataPoolBytes = std::size_t{1} << 30;      // 1 GiB
inline constexpr std::size_t kDefaultMetadataPoolBytes = std::size_t{1} << 26;  // 64 MiB

/// How DeviceContext backs dataResource()/metadataResource(). Both
/// alternatives are `cuda::mr::any_resource<cuda::mr::device_accessible>`
/// underneath (CCCL's own type-erased memory_resource wrapper) so
/// StitchPool -- and anything else built against DeviceContext -- calls the
/// exact same allocate()/deallocate() surface regardless of which is
/// active; adding a third strategy later (e.g. an async/stream-ordered
/// pool) only touches MakeResource() in device_context.cpp.
enum class AllocationPolicy {
	/// Reserve one big arena per pool up front (see kDefaultDataPoolBytes /
	/// kDefaultMetadataPoolBytes) and suballocate from it
	/// (rmm::mr::pool_memory_resource, coalescing best-fit) -- Arachne
	/// manages the pool itself; cudaMalloc/cudaFree happen rarely, only when
	/// a pool's current reservation is exhausted. This is the
	/// fragmentation-manageable policy StitchPool::compact() is meant to
	/// operate on.
	Pooled,
	/// No pre-reservation, no suballocation: every StitchPool::allocate()
	/// issues its own independent cudaMalloc, and free() its own cudaFree
	/// (rmm::mr::cuda_memory_resource directly). Simplest possible strategy,
	/// and the natural baseline/fallback -- fragmentation here is the CUDA
	/// driver's own allocator's problem, entirely outside Arachne's control
	/// either way, so there is nothing for StitchPool::compact() to do under
	/// this policy.
	Naive,
};

/// Owns Arachne's CUDA device selection, the RAFT resources handle (default
/// stream, stream pool, and lazily-created cuBLAS/cuSOLVER/cuSPARSE handles
/// -- see raft::device_resources), and the two device memory resources
/// every gpu::StitchPool allocates against (see MemoryKind in
/// gpu/stitch_handle.hpp):
///
///  - dataResource(): Anchor-driven, promote/evict-eligible index/vector-
///    data memory (MemoryKind::Data).
///  - metadataResource(): a physically separate resource for state that
///    needs CPU/GPU sync but must never be evicted the way Stitch data can
///    be (MemoryKind::Metadata).
///
/// Both are constructed according to `policy` (see AllocationPolicy) from
/// the same raw cudaMalloc/cudaFree upstream (memoryResource()), and stay
/// independent of each other either way, so fragmentation/pressure in one
/// never affects the other.
///
/// One DeviceContext per physical GPU. Owned by Controller (see
/// core/controller.hpp) -- Controller is already the class that owns every
/// other piece of Arachne's own policy state (AnchorManager, OpScheduler),
/// and GPU residency accounting is exactly that kind of state, not a
/// pluggable dependency like IndexAdapter/RoutingCache.
///
/// Multi-GPU sharding is future work -- StitchPool's callers don't need to
/// change for it.
class DeviceContext {
 public:
	explicit DeviceContext(int device_id = 0, AllocationPolicy policy = AllocationPolicy::Pooled,
												 std::size_t data_pool_bytes = kDefaultDataPoolBytes,
												 std::size_t metadata_pool_bytes = kDefaultMetadataPoolBytes);

	DeviceContext(const DeviceContext&) = delete;
	DeviceContext& operator=(const DeviceContext&) = delete;

	int deviceId() const { return device_id_; }
	AllocationPolicy allocationPolicy() const { return policy_; }

	raft::device_resources& resources() { return resources_; }
	const raft::device_resources& resources() const { return resources_; }

	/// The raw cudaMalloc/cudaFree resource dataResource()/metadataResource()
	/// are themselves built from (directly, under Naive; as a
	/// pool_memory_resource's upstream, under Pooled). Not meant for
	/// StitchPool to allocate against directly -- see MemoryKind's doc
	/// comment.
	rmm::mr::cuda_memory_resource& memoryResource() { return memory_resource_; }

	cuda::mr::any_resource<cuda::mr::device_accessible>& dataResource() { return data_resource_; }
	cuda::mr::any_resource<cuda::mr::device_accessible>& metadataResource() { return metadata_resource_; }

 private:
	// Declaration order matters: device_id_'s initializer (see
	// device_context.cpp) is what actually calls cudaSetDevice(), and it must
	// run before resources_ default-constructs its stream/handles, and before
	// data_resource_/metadata_resource_ potentially make a real, immediate
	// cudaMalloc call (Pooled: reserving their initial arena; Naive: none at
	// construction time) -- member initialization runs in declaration order
	// regardless of the constructor's init-list order, so device_id_ must
	// stay the first member declared here, and the resources must stay after
	// memory_resource_ (their upstream).
	int device_id_;
	AllocationPolicy policy_;
	raft::device_resources resources_;
	rmm::mr::cuda_memory_resource memory_resource_;
	cuda::mr::any_resource<cuda::mr::device_accessible> data_resource_;
	cuda::mr::any_resource<cuda::mr::device_accessible> metadata_resource_;
};

}  // namespace arachne::gpu
