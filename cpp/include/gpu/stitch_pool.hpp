#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "gpu/device_context.hpp"
#include "gpu/stitch_handle.hpp"

namespace arachne::gpu {

/// Arachne-owned GPU memory allocator backing every Stitch (see
/// core/anchor_manager.hpp). Per the Anchor-centric residency policy:
/// promote/evict decisions are made per-Stitch based on the owning
/// Anchor's observed hotness/latency/transfer-cost, not on fixed-size
/// address-based pages -- so allocations here are variable-sized and
/// identified by opaque handle, not laid out on any fixed slab grid.
/// Adapters never call cudaMalloc directly for memory a Stitch is meant to
/// cover; they go through the StitchPool threaded down from Controller so
/// Arachne, not the index, accounts for (and eventually migrates)
/// residency.
///
/// Backed by DeviceContext's two pre-allocated pools (dataPool()/
/// metadataPool(), routed to by `kind` -- see MemoryKind), not by raw
/// cudaMalloc/cudaFree per call: DeviceContext reserves each pool's arena
/// once up front, so a StitchPool allocate()/free() only round-trips to
/// cudaMalloc/cudaFree when a pool's current reservation is actually
/// exhausted.
///
/// Thread-safe: allocate()/access()/free() each take their own lock,
/// mirroring AnchorManager's own concurrency contract (Controller is called
/// concurrently the same way AnchorManager is).
class StitchPool {
 public:
	explicit StitchPool(DeviceContext& device);
	~StitchPool();

	StitchPool(const StitchPool&) = delete;
	StitchPool& operator=(const StitchPool&) = delete;

	/// Allocates `bytes` of device memory from the pool matching `kind` (see
	/// MemoryKind) and returns a handle to it.
	StitchHandle allocate(std::size_t bytes, MemoryKind kind = MemoryKind::Data);

	/// Resolves `handle` to its device pointer. Throws std::invalid_argument
	/// if `handle` was never returned by allocate() on this pool, or was
	/// already freed.
	void* access(StitchHandle handle, AccessMode mode);

	/// Releases the allocation backing `handle`. No-op if `handle` is
	/// invalid or was already freed.
	void free(StitchHandle handle);

	/// Total bytes currently outstanding across all live allocations of both
	/// kinds -- the sizing signal the eventual Anchor-driven promote/evict
	/// policy needs to reason about a GPU memory budget (that policy only
	/// ever concerns MemoryKind::Data, hence the per-kind overload below).
	std::size_t bytesAllocated() const;

	/// Bytes currently outstanding for just `kind`.
	std::size_t bytesAllocated(MemoryKind kind) const;

 private:
	struct Allocation {
		void* device_ptr = nullptr;
		std::size_t bytes = 0;
		MemoryKind kind = MemoryKind::Data;
	};

	rmm::mr::pool_memory_resource& poolFor(MemoryKind kind);

	DeviceContext& device_;
	mutable std::mutex mutex_;
	std::unordered_map<std::uint64_t, Allocation> allocations_;
	std::uint64_t next_id_ = 1;
};

}  // namespace arachne::gpu
