#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <cuda/memory_resource>
#include <cuda_runtime.h>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include "gpu/device_region_handle.hpp"
#include "gpu/unit_pool_arena.hpp"

namespace arachne::gpu {

/// Placeholder budget sizes pending real tuning against actual GPU memory
/// (e.g. via cudaMemGetInfo) and workload characteristics -- see the class
/// overview below for how AllocationPolicy uses them.
inline constexpr std::size_t kDefaultDataPoolBytes = std::size_t{1} << 30;      // 1 GiB
inline constexpr std::size_t kDefaultMetadataPoolBytes = std::size_t{1} << 26;  // 64 MiB

/// Default accounting/allocation unit for AllocationPolicy::Pooled. A Region
/// remains the semantic admission, transfer, and eviction object; this 4 KiB
/// unit only determines how finely the preallocated arena accounts for its
/// contiguous storage. Keeping those concepts separate lets future partial-
/// residency policies select Region-owned extents without turning individual
/// pages into independently replaceable cache objects. Callers can still
/// choose a larger unit for lower allocator-metadata overhead.
inline constexpr std::size_t kDefaultUnitBytes = std::size_t{1} << 12;  // 4 KiB

/// How DeviceContext backs dataArena()/metadataArena()/dataResource()/
/// metadataResource() -- see the class overview below. Adding a third
/// strategy later only touches DeviceContext's constructor.
enum class AllocationPolicy {
	Pooled,
	Async,
	Normal = Async,
};

/// DeviceContext owns everything physical about one GPU: CUDA device
/// selection, the RAFT resources handle (default stream, stream pool, and
/// lazily-created cuBLAS/cuSOLVER/cuSPARSE handles -- see
/// raft::device_resources), and the device memory backing every
/// gpu::DeviceRegionPool allocation (see MemoryKind in
/// gpu/device_region_handle.hpp).
///
/// One DeviceContext per physical GPU, owned by Controller (see
/// core/controller.hpp) alongside every other piece of Arachne's own policy
/// state (RegionManager, OpScheduler) -- GPU residency accounting is
/// exactly that kind of state, not a pluggable dependency like
/// IAdapter/RoutingCache. Multi-GPU sharding is future work;
/// DeviceRegionPool's callers don't need to change for it.
///
/// Two independent memory pools, routed to by MemoryKind:
///  - Data: Anchor-driven, promote/evict-eligible index/vector-data memory.
///  - Metadata: a physically separate resource for state that needs
///    CPU/GPU sync but must never be evicted the way Region data can be.
/// Both are constructed according to `policy` from the same raw
/// cudaMalloc/cudaFree upstream (memoryResource()), and stay independent of
/// each other either way, so fragmentation/pressure in one never affects
/// the other.
///
/// AllocationPolicy selects how each pool is backed:
///
///   AllocationPolicy::Normal                 AllocationPolicy::Pooled
///   -----------------------------------      ----------------------------
///   No pre-reservation, no                   One big arena reserved up
///   suballocation: every                     front per pool (see
///   DeviceRegionPool::allocate() issues       kDefaultDataPoolBytes /
///   its own independent cudaMalloc,           kDefaultMetadataPoolBytes),
///   and free() its own cudaFree,              fixed-size-unit-managed
///   straight through dataResource()/          (UnitPoolArena). cudaMalloc/
///   metadataResource()                        cudaFree happen exactly once
///   (rmm::mr::cuda_memory_resource             per pool, at construction/
///   directly). Simplest possible               destruction, and the arena
///   strategy, and the natural                  never grows afterward: a
///   baseline/fallback.                         request the budget can't
///                                              satisfy is a real capacity/
///   Fragmentation here is the CUDA             fragmentation event, not
///   driver's own allocator's                   something silently absorbed
///   problem, entirely outside                  by an upstream cudaMalloc
///   Arachne's control either way --            the way the old
///   nothing for                                rmm::mr::pool_memory_resource
///   DeviceRegionPool::compact() to             -backed design worked. This
///   do, and no arena for it to even             is the fragmentation-
///   operate against.                           manageable policy compact()
///                                              (and its injected
///                                              CompactionPolicy) operates
///                                              on.
///
/// budgetBytes(kind) is Arachne's own self-imposed ceiling on how much
/// `kind` memory it is willing to have resident: under Pooled, the actual
/// unit-rounded arena capacity; under Normal, exactly the configured pool
/// size, self-capped even though the CUDA driver could technically grant
/// more, so promotion/eviction decisions stay deterministic and don't
/// depend on racing actual GPU memory pressure. It is purely the number
/// this DeviceContext was configured with, not a live query against GPU
/// state (see cudaMemGetInfo for that).
///
/// Streams: managementStream() is raft::device_resources' own canonical
/// stream, used for Arachne's own residency management (promotion,
/// eviction/write-back, compaction -- see core/region_manager.hpp's
/// Coordinator). workerStream(index) is a separate, dedicated stream per
/// OpScheduler execution worker, so concurrent workers' GPU-native
/// traverseDevice()/modifyDevice() kernel launches can genuinely overlap
/// instead of serializing behind one shared stream. The two are kept
/// separate so management traffic never queues behind (or in front of) a
/// worker's own kernel launches -- see gpu::DeviceRegionPool::acquire()'s
/// cross-stream event-wait for how they're kept safely ordered against
/// each other despite being physically different streams.
class DeviceContext {
 public:
	// `worker_stream_count` sizes the workerStream() pool (callers pass
	// SchedulingConfig::max_execution_threads for one stream per worker).
	// `unit_bytes` sizes the Pooled arena's fixed subregion unit; ignored
	// under AllocationPolicy::Normal.
	explicit DeviceContext(int device_id = 0, AllocationPolicy policy = AllocationPolicy::Async,
												 std::size_t data_pool_bytes = kDefaultDataPoolBytes,
												 std::size_t metadata_pool_bytes = kDefaultMetadataPoolBytes,
												 std::size_t worker_stream_count = 1, std::size_t unit_bytes = kDefaultUnitBytes);
	~DeviceContext();

	DeviceContext(const DeviceContext&) = delete;
	DeviceContext& operator=(const DeviceContext&) = delete;

	int deviceId() const { return device_id_; }
	AllocationPolicy allocationPolicy() const { return policy_; }

	/// Arachne's self-imposed ceiling for `kind` memory (enforced by
	/// gpu::DeviceRegionPool::hasCapacity()/tryAllocate()) -- see the class
	/// overview above for how this differs between Pooled (actual,
	/// unit-rounded arena capacity) and Normal (exactly the configured pool
	/// size).
	std::size_t budgetBytes(MemoryKind kind) const;

	raft::device_resources& resources() { return resources_; }
	const raft::device_resources& resources() const { return resources_; }

	/// raft::device_resources' own canonical stream, exposed under this more
	/// specific name now that workerStream() exists as its compute-side
	/// counterpart -- see the class overview above for why the two are kept
	/// separate.
	cudaStream_t managementStream() const { return resources_.get_stream().value(); }

	/// One dedicated stream per OpScheduler execution worker (see
	/// core/op_scheduler.hpp's SchedulingConfig::max_execution_threads) --
	/// see the class overview above. `index` is 0-based and must be <
	/// workerStreamCount(); throws std::out_of_range otherwise.
	cudaStream_t workerStream(std::size_t index) const;

	std::size_t workerStreamCount() const { return worker_streams_.size(); }

	/// One dedicated, persistent scratch buffer per OpScheduler execution
	/// worker -- same "index -> fixed resource" pattern as workerStream(),
	/// for GPU memory instead of a stream. Meant for an adapter's own
	/// short-lived per-call/per-hop device buffers (see IAdapter::
	/// requiredScratchBytesPerWorker()'s doc comment for the intended use and
	/// core/controller.hpp's Controller::workerScratch() for how a worker
	/// thread reaches it), so a hot loop like HnswlibIndexGpu::
	/// TraverseBatchOnDevice()'s per-hop distance computation doesn't pay a
	/// cudaMalloc/cudaFree every round. Bypasses the Pooled/Async arena
	/// machinery entirely -- unlike Region data, this is allocated exactly
	/// once per DeviceContext lifetime and never grows/shrinks/fragments, so
	/// there's nothing for that machinery to do here.
	///
	/// `index` must be < workerStreamCount(). Returns nullptr if
	/// reserveWorkerScratch() was never called, or was called with
	/// `bytes_per_worker == 0` -- a caller must treat that as "no scratch
	/// available" and fall back to its own allocation, not as an error.
	void* workerScratch(std::size_t index) const;

	std::size_t workerScratchBytesPerWorker() const { return worker_scratch_bytes_; }

	/// Reserves workerStreamCount() * `bytes_per_worker` bytes in one
	/// cudaMalloc, sliced evenly across workerScratch(0..workerStreamCount()-1).
	/// Must be called at most once, and before any OpScheduler worker thread
	/// starts (Controller's constructor does this, right before
	/// OpScheduler::start()) -- throws std::logic_error if called a second
	/// time. `bytes_per_worker == 0` is a valid, cheap no-op: no cudaMalloc
	/// happens, and every workerScratch(index) call keeps returning nullptr.
	void reserveWorkerScratch(std::size_t bytes_per_worker);

	/// The raw cudaMalloc/cudaFree resource both pools are ultimately backed
	/// by (see the class overview above). Not meant for DeviceRegionPool to
	/// allocate against directly -- see MemoryKind's doc comment.
	rmm::mr::cuda_memory_resource& memoryResource() { return memory_resource_; }

	/// AllocationPolicy::Normal's direct allocate()/deallocate() surface.
	/// Always constructed (harmless if unused under Pooled), so switching
	/// `policy` never changes which accessors are valid to call.
	cuda::mr::any_resource<cuda::mr::device_accessible>& dataResource() { return data_resource_; }
	cuda::mr::any_resource<cuda::mr::device_accessible>& metadataResource() { return metadata_resource_; }

	/// Non-null iff allocationPolicy() == Pooled -- see UnitPoolArena's own
	/// doc comment for what these actually provide over dataResource()/
	/// metadataResource().
	UnitPoolArena* dataArena() { return data_arena_.get(); }
	UnitPoolArena* metadataArena() { return metadata_arena_.get(); }
	const UnitPoolArena* dataArena() const { return data_arena_.get(); }
	const UnitPoolArena* metadataArena() const { return metadata_arena_.get(); }

 private:
	// Declaration order matters: member init runs in declaration order (not
	// constructor init-list order), and device_id_'s initializer calls
	// cudaSetDevice(), which must happen before resources_ and the
	// resource/arena members below construct.
	int device_id_;
	AllocationPolicy policy_;
	std::size_t data_pool_bytes_;
	std::size_t metadata_pool_bytes_;
	raft::device_resources resources_;
	rmm::mr::cuda_memory_resource memory_resource_;
	cuda::mr::any_resource<cuda::mr::device_accessible> data_resource_;
	cuda::mr::any_resource<cuda::mr::device_accessible> metadata_resource_;
	// Non-null only under AllocationPolicy::Pooled -- constructed in the
	// constructor body (after data_resource_/metadata_resource_, which they
	// preallocate their one big buffer from), destroyed by the destructor.
	std::unique_ptr<UnitPoolArena> data_arena_;
	std::unique_ptr<UnitPoolArena> metadata_arena_;
	// Created in the constructor body (after device_id_'s cudaSetDevice()
	// has already run), destroyed in the destructor -- see workerStream()'s
	// doc comment.
	std::vector<cudaStream_t> worker_streams_;

	// Set by reserveWorkerScratch() (at most once); worker_scratch_base_
	// stays null until then, or forever if bytes_per_worker was 0 -- see
	// workerScratch()'s doc comment.
	std::size_t worker_scratch_bytes_ = 0;
	void* worker_scratch_base_ = nullptr;
};

}  // namespace arachne::gpu
