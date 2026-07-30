#pragma once

#include <cstddef>
#include <vector>

#include <cuda/memory_resource>
#include <cuda_runtime.h>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include "gpu/device_region_handle.hpp"

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
/// DeviceRegionPool -- and anything else built against DeviceContext -- calls the
/// exact same allocate()/deallocate() surface regardless of which is
/// active; adding a third strategy later (e.g. an async/stream-ordered
/// pool) only touches MakeResource() in device_context.cpp.
enum class AllocationPolicy {
	/// Reserve one big arena per pool up front (see kDefaultDataPoolBytes /
	/// kDefaultMetadataPoolBytes) and suballocate from it
	/// (rmm::mr::pool_memory_resource, coalescing best-fit) -- Arachne
	/// manages the pool itself; cudaMalloc/cudaFree happen rarely, only when
	/// a pool's current reservation is exhausted. This is the
	/// fragmentation-manageable policy DeviceRegionPool::compact() is meant to
	/// operate on.
	Pooled,
	/// No pre-reservation, no suballocation: every DeviceRegionPool::allocate()
	/// issues its own independent cudaMalloc, and free() its own cudaFree
	/// (rmm::mr::cuda_memory_resource directly). Simplest possible strategy,
	/// and the natural baseline/fallback -- fragmentation here is the CUDA
	/// driver's own allocator's problem, entirely outside Arachne's control
	/// either way, so there is nothing for DeviceRegionPool::compact() to do under
	/// this policy.
	Naive,
};

/// Owns Arachne's CUDA device selection, the RAFT resources handle (default
/// stream, stream pool, and lazily-created cuBLAS/cuSOLVER/cuSPARSE handles
/// -- see raft::device_resources), and the two device memory resources
/// every gpu::DeviceRegionPool allocates against (see MemoryKind in
/// gpu/device_region_handle.hpp):
///
///  - dataResource(): Anchor-driven, promote/evict-eligible index/vector-
///    data memory (MemoryKind::Data).
///  - metadataResource(): a physically separate resource for state that
///    needs CPU/GPU sync but must never be evicted the way Region data can
///    be (MemoryKind::Metadata).
///
/// Both are constructed according to `policy` (see AllocationPolicy) from
/// the same raw cudaMalloc/cudaFree upstream (memoryResource()), and stay
/// independent of each other either way, so fragmentation/pressure in one
/// never affects the other.
///
/// One DeviceContext per physical GPU. Owned by Controller (see
/// core/controller.hpp) -- Controller is already the class that owns every
/// other piece of Arachne's own policy state (RegionManager, OpScheduler),
/// and GPU residency accounting is exactly that kind of state, not a
/// pluggable dependency like IAdapter/RoutingCache.
///
/// Multi-GPU sharding is future work -- DeviceRegionPool's callers don't need to
/// change for it.
class DeviceContext {
 public:
	// `worker_stream_count` sizes the workerStream() pool below -- callers
	// (Controller) pass SchedulingConfig::max_execution_threads so there's
	// exactly one dedicated stream per OpScheduler execution worker.
	explicit DeviceContext(int device_id = 0, AllocationPolicy policy = AllocationPolicy::Pooled,
												 std::size_t data_pool_bytes = kDefaultDataPoolBytes,
												 std::size_t metadata_pool_bytes = kDefaultMetadataPoolBytes,
												 std::size_t worker_stream_count = 1);
	~DeviceContext();

	DeviceContext(const DeviceContext&) = delete;
	DeviceContext& operator=(const DeviceContext&) = delete;

	int deviceId() const { return device_id_; }
	AllocationPolicy allocationPolicy() const { return policy_; }

	/// The self-imposed ceiling gpu::DeviceRegionPool::hasCapacity()/
	/// tryAllocate() enforce for `kind` (data_pool_bytes/metadata_pool_bytes
	/// as passed to the constructor, or the defaults above) -- this is
	/// Arachne's own accounting of how much `kind` memory it is willing to
	/// have resident at once, independent of whether the allocator backing
	/// it physically pre-reserves that much (Pooled) or not (Naive, where
	/// nothing stops the CUDA driver from granting more; Arachne caps itself
	/// anyway so promotion/eviction decisions are deterministic and don't
	/// depend on racing actual GPU memory pressure). Not a query against
	/// live GPU state (see cudaMemGetInfo for that) -- purely the number
	/// this DeviceContext was configured with.
	std::size_t budgetBytes(MemoryKind kind) const {
		return kind == MemoryKind::Metadata ? metadata_pool_bytes_ : data_pool_bytes_;
	}

	raft::device_resources& resources() { return resources_; }
	const raft::device_resources& resources() const { return resources_; }

	/// The stream Arachne's own GPU-residency management (promotion,
	/// eviction/write-back, compaction -- see core/region_manager.hpp's
	/// Coordinator) issues its data movement on, deliberately kept separate
	/// from workerStream() below so management traffic never queues behind
	/// (or in front of) a worker's own kernel launches on the same stream --
	/// see gpu::DeviceRegionPool::acquire()'s cross-stream event-wait for how
	/// the two are kept safely ordered relative to each other despite being
	/// physically different streams. This is raft::device_resources' own
	/// canonical stream, exposed under this more specific name now that
	/// workerStream() exists as its compute-side counterpart.
	cudaStream_t managementStream() const { return resources_.get_stream().value(); }

	/// One dedicated stream per OpScheduler execution worker (see
	/// core/op_scheduler.hpp's SchedulingConfig::max_execution_threads), so
	/// concurrent worker threads' GPU-native traverseDevice()/modifyDevice()
	/// kernel launches can genuinely overlap on the GPU instead of
	/// serializing behind one shared stream the way they would if every
	/// caller defaulted to managementStream(). `index` is 0-based and must be
	/// < workerStreamCount() -- throws std::out_of_range otherwise.
	cudaStream_t workerStream(std::size_t index) const;

	std::size_t workerStreamCount() const { return worker_streams_.size(); }

	/// The raw cudaMalloc/cudaFree resource dataResource()/metadataResource()
	/// are themselves built from (directly, under Naive; as a
	/// pool_memory_resource's upstream, under Pooled). Not meant for
	/// DeviceRegionPool to allocate against directly -- see MemoryKind's doc
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
	std::size_t data_pool_bytes_;
	std::size_t metadata_pool_bytes_;
	raft::device_resources resources_;
	rmm::mr::cuda_memory_resource memory_resource_;
	cuda::mr::any_resource<cuda::mr::device_accessible> data_resource_;
	cuda::mr::any_resource<cuda::mr::device_accessible> metadata_resource_;
	// Created in the constructor body (after device_id_'s cudaSetDevice()
	// has already run), destroyed in the destructor -- see workerStream()'s
	// doc comment.
	std::vector<cudaStream_t> worker_streams_;
};

}  // namespace arachne::gpu
