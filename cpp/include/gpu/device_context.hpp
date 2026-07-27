#pragma once

#include <cstddef>

#include <raft/core/device_resources.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace arachne::gpu {

/// Reserved up front, once, at DeviceContext construction -- see
/// dataPool()/metadataPool() and their doc comments below. Each is only the
/// *initial* reservation; the underlying rmm::mr::pool_memory_resource
/// still grows on demand (coalescing more from the upstream
/// cuda_memory_resource) if a pool's allocations exceed it, so these are
/// sizing hints to avoid early cudaMalloc round-trips, not hard caps.
/// Placeholder values pending real budget tuning against actual GPU memory
/// (e.g. via cudaMemGetInfo) and workload characteristics.
inline constexpr std::size_t kDefaultDataPoolBytes = std::size_t{1} << 30;    // 1 GiB
inline constexpr std::size_t kDefaultMetadataPoolBytes = std::size_t{1} << 26;  // 64 MiB

/// Owns Arachne's CUDA device selection, the RAFT resources handle (default
/// stream, stream pool, and lazily-created cuBLAS/cuSOLVER/cuSPARSE handles
/// -- see raft::device_resources), and the two device memory pools every
/// gpu::StitchPool allocates against (see MemoryKind in gpu/stitch_handle.hpp):
///
///  - dataPool(): Anchor-driven, promote/evict-eligible index/vector-data
///    memory (MemoryKind::Data).
///  - metadataPool(): a physically separate pool for state that needs
///    CPU/GPU sync but must never be evicted the way Stitch data can be
///    (MemoryKind::Metadata).
///
/// Both are backed by the same raw cudaMalloc/cudaFree upstream
/// (memoryResource()) but reserve their own initial arena on construction
/// (see kDefaultDataPoolBytes/kDefaultMetadataPoolBytes) and coalesce
/// within themselves independently, so fragmentation in one never affects
/// the other, and repeated allocate()/free() calls on a StitchPool don't
/// round-trip to cudaMalloc/cudaFree once each pool's initial reservation
/// covers the request.
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
	explicit DeviceContext(int device_id = 0, std::size_t data_pool_bytes = kDefaultDataPoolBytes,
												 std::size_t metadata_pool_bytes = kDefaultMetadataPoolBytes);

	DeviceContext(const DeviceContext&) = delete;
	DeviceContext& operator=(const DeviceContext&) = delete;

	int deviceId() const { return device_id_; }

	raft::device_resources& resources() { return resources_; }
	const raft::device_resources& resources() const { return resources_; }

	/// The raw cudaMalloc/cudaFree resource dataPool()/metadataPool() are
	/// themselves carved from. Not meant for StitchPool to allocate against
	/// directly -- see MemoryKind's doc comment.
	rmm::mr::cuda_memory_resource& memoryResource() { return memory_resource_; }

	rmm::mr::pool_memory_resource& dataPool() { return data_pool_; }
	rmm::mr::pool_memory_resource& metadataPool() { return metadata_pool_; }

 private:
	// Declaration order matters: device_id_'s initializer (see
	// device_context.cpp) is what actually calls cudaSetDevice(), and it must
	// run before resources_ default-constructs its stream/handles, and before
	// data_pool_/metadata_pool_ make their real, immediate cudaMalloc calls to
	// reserve their initial arenas -- member initialization runs in
	// declaration order regardless of the constructor's init-list order, so
	// device_id_ must stay the first member declared here, and the pools must
	// stay after memory_resource_ (their upstream).
	int device_id_;
	raft::device_resources resources_;
	rmm::mr::cuda_memory_resource memory_resource_;
	rmm::mr::pool_memory_resource data_pool_;
	rmm::mr::pool_memory_resource metadata_pool_;
};

}  // namespace arachne::gpu
