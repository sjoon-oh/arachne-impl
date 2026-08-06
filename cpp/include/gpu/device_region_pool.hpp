#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <unordered_map>

#include <cuda/memory_resource>
#include <cuda_runtime.h>

#include "gpu/compaction_policy.hpp"
#include "gpu/device_context.hpp"
#include "gpu/device_memory_manager.hpp"
#include "gpu/device_region_handle.hpp"
#include "gpu/pinned_host_pool.hpp"
#include "gpu/unit_pool_arena.hpp"

namespace arachne::gpu {

/// Arachne-owned GPU memory allocator backing every Region (see
/// core/region_manager.hpp). Per the Anchor-centric residency policy,
/// promote/evict decisions are made per-Region based on the depending
/// Anchors' observed hotness/latency/transfer-cost, not on fixed-size
/// address-based pages -- so allocations here are variable-sized and
/// identified by opaque handle, not laid out on any fixed slab grid.
/// Adapters never call cudaMalloc directly for memory a Region is meant to
/// cover; they go through the DeviceRegionPool threaded down from Controller
/// so Arachne, not the index, accounts for (and eventually migrates)
/// residency.
///
/// Backed by DeviceContext, routed to by `kind` (see MemoryKind), one of two
/// ways depending on DeviceContext::allocationPolicy() (see AllocationPolicy
/// in gpu/device_context.hpp):
///
///   AllocationPolicy::Normal               AllocationPolicy::Pooled
///   allocate()/free() go straight          allocate()/free() go through
///   through to DeviceContext's             DeviceContext's dataArena()/
///   dataResource()/metadataResource()      metadataArena() (a UnitPoolArena:
///   -- a real cudaMalloc/cudaFree          one big preallocated buffer,
///   each call.                            suballocated in fixed-size units).
///                                         Fragmentation-triggered relocation
///                                         is handled by an injected
///                                         CompactionPolicy (see compact()).
///
/// The two policies no longer share one uniform allocate()/deallocate()
/// surface -- DeviceRegionPool itself branches on allocationPolicy() at each
/// entry point (allocateNormal()/allocatePooled(), freeNormal()/
/// freePooled()) precisely because Pooled needs the richer, relocation-aware
/// UnitPoolArena surface that a generic cuda::mr resource can't provide (see
/// UnitPoolArena's and CompactionPolicy's own doc comments for why).
///
/// Lease and cross-stream safety: a raw device pointer is never handed out
/// directly. acquire() resolves a handle to a pointer only inside a Lease,
/// which marks the allocation "in use" until released. compact() and free()
/// both wait for every outstanding Lease on a handle to be released -- and
/// for the GPU to actually catch up to that release point -- before
/// relocating or reclaiming its memory. Because CUDA work is asynchronous,
/// "released" alone doesn't mean "GPU is done": release() records a CUDA
/// event on the Lease's stream marking everything enqueued up to that point,
/// and a future compact()/free() call waits on that event via
/// cudaStreamWaitEvent (not a host-blocking sync), so the common case (no
/// pending work) costs nothing. This same event-wait discipline is what lets
/// acquire() safely hand different callers (per-worker compute streams, the
/// management stream -- see DeviceContext) genuinely different streams over
/// the same allocation: before returning a Lease, acquire() inserts a
/// GPU-side cudaStreamWaitEvent on the requested stream for every other
/// stream's still-pending "last used" event, so work on the new stream is
/// guaranteed ordered-after any prior use on a different stream.
///
/// Compaction executor (tryOpenContiguousExtentLocked(), shared by
/// allocatePooled()'s internal retry and compact()'s explicit call): under
/// Pooled, when `kind`'s largest free extent can't satisfy required_units
/// but totalFreeUnits() can, this snapshots every currently-unpinned live
/// allocation of `kind` into a MovableBlock list, asks compaction_policy_
/// for a Plan against the arena's current free-extent state, then executes
/// it move by move -- for each Move: re-validates the block is still
/// present and unpinned (cheap insurance; mutex_ has been held continuously
/// since the snapshot, so nothing can actually have changed), waits for it
/// to go quiescent (awaitQuiescentLocked(), guaranteed non-blocking here
/// since only already-unpinned blocks were ever offered), claims the
/// destination range, issues the D2D copy on DeviceContext's canonical
/// stream, updates the allocation's device_ptr/unit_range, and frees the
/// vacated source range. Returns {} immediately, without ever consulting
/// compaction_policy_, if the largest free extent already satisfies
/// required_units, or if totalFreeUnits() doesn't either (a genuine
/// capacity shortfall no relocation plan can fix).
///
/// allocatePooled() invokes this internally, targeted at the one request
/// that just failed best-fit, so an ordinary fragmentation-induced
/// allocation failure self-heals without every caller needing to know to
/// call compact() themselves; compact() invokes the same mechanism targeted
/// at "everything the policy is willing to merge" as its own explicit,
/// caller-triggered entry point. Either way, only allocations with
/// in_use_count == 0 at snapshot time are ever relocated -- a pinned
/// allocation is simply never offered to the CompactionPolicy, never
/// something either path blocks waiting to become unpinned the way an older
/// unconditional-relocate-everything design did. Cost is bounded by the
/// injected CompactionPolicy's own budget (see
/// TargetedCompactionPolicy::Budget), never "every live allocation of
/// `kind`".
///
/// Thread-safe: every method takes DeviceRegionPool's own lock (mutex_),
/// mirroring RegionManager's own concurrency contract (Controller is called
/// concurrently the same way RegionManager is).
class DeviceRegionPool {
 public:
	/// RAII marker: the current holder is actively using `handle`'s device
	/// memory (work may still be in flight on stream()). While alive,
	/// DeviceRegionPool guarantees `handle` will not be relocated or
	/// reclaimed out from under it -- see the class overview above for how
	/// release() and the event-wait mechanism make that safe without a host
	/// block. Move-only: "currently in use" ownership is unique per Lease.
	class Lease {
	 public:
		~Lease();

		Lease(Lease&& other) noexcept;
		Lease& operator=(Lease&& other) noexcept;
		Lease(const Lease&) = delete;
		Lease& operator=(const Lease&) = delete;

		/// The handle's device pointer, resolved at acquire() time and stable
		/// for this Lease's lifetime (nothing relocates/frees a handle with an
		/// outstanding Lease) -- safe to hold onto and pass into an
		/// asynchronous kernel launch on stream() for as long as the Lease is
		/// alive.
		void* ptr() const { return ptr_; }

		cudaStream_t stream() const { return stream_; }

	 private:
		friend class DeviceRegionPool;
		Lease(DeviceRegionPool& pool, DeviceRegionHandle handle, cudaStream_t stream, void* ptr);

		DeviceRegionPool* pool_;  // null after being moved from
		DeviceRegionHandle handle_;
		cudaStream_t stream_;
		void* ptr_;
	};

	struct TransferBatch {
		std::vector<Lease> leases;
		std::vector<PinnedHostPool::Buffer> pinned_sources;
	};

	/// `compaction_policy` governs compact() and allocate()'s internal
	/// fragmentation retry under Pooled; null defaults to
	/// TargetedCompactionPolicy with its default Budget. Safe to pass one
	/// under Normal too -- it's simply never invoked there.
	explicit DeviceRegionPool(DeviceContext& device, std::unique_ptr<CompactionPolicy> compaction_policy = nullptr);
	~DeviceRegionPool();

	DeviceRegionPool(const DeviceRegionPool&) = delete;
	DeviceRegionPool& operator=(const DeviceRegionPool&) = delete;

	/// Allocates `bytes` of `kind` memory and returns a handle. Throws if the
	/// allocation fails; callers needing a non-throwing, capacity-aware
	/// attempt (Controller's Promotion path) should use tryAllocate() instead.
	///
	/// Under Pooled, a best-fit failure triggers exactly one internal
	/// fragmentation-relocation attempt (tryOpenContiguousExtentLocked(),
	/// targeted at this request's own size) before retrying once and, only
	/// then, throwing std::runtime_error -- see the class overview above for
	/// why this self-heals without callers needing to call compact()
	/// themselves.
	DeviceRegionHandle allocate(std::size_t bytes, MemoryKind kind = MemoryKind::Data);

	/// True if allocating `bytes` more of `kind` would stay within
	/// DeviceContext::budgetBytes(kind), checked against
	/// bytesAllocated(kind). A snapshot, not a reservation -- see
	/// tryAllocate()'s doc comment for why the resulting race is acceptable.
	bool hasCapacity(std::size_t bytes, MemoryKind kind = MemoryKind::Data) const;

	/// Capacity-aware allocate(): returns std::nullopt instead of throwing,
	/// whether hasCapacity() already said no or the underlying allocate()
	/// throws anyway (e.g. real GPU pressure under Normal, or an uncloseable
	/// Pooled fragmentation shortfall). Lets callers (e.g.
	/// Controller::make()) treat "no room right now" as retryable rather
	/// than exceptional.
	///
	/// The race hasCapacity() leaves open -- concurrent callers both seeing
	/// headroom, then both allocating and together overshooting budget -- is
	/// accepted rather than closed with a reservation step: callers act on
	/// the result immediately, and a bounded overshoot is cheaper than
	/// serializing every allocate() behind a second lock acquisition.
	std::optional<DeviceRegionHandle> tryAllocate(std::size_t bytes, MemoryKind kind = MemoryKind::Data);

	/// Marks `handle` in use on `stream` until the returned Lease is
	/// destroyed -- the only way to resolve a handle to its device pointer
	/// (see Lease's and the class overview's own doc comments for why a bare
	/// pointer-returning accessor doesn't exist anymore). Throws
	/// std::invalid_argument if `handle` was never returned by allocate() on
	/// this pool, or was already freed.
	Lease acquire(DeviceRegionHandle handle, cudaStream_t stream);

	/// Same as the two-argument overload, on DeviceContext's own management
	/// stream (see DeviceContext::managementStream()).
	Lease acquire(DeviceRegionHandle handle);

	/// Releases the allocation backing `handle`. No-op if invalid or already
	/// freed. Blocks until every outstanding Lease has been released and the
	/// GPU has caught up (see Lease's doc comment) -- callers should expect
	/// this to wait rather than fail or skip an already-chosen eviction
	/// target (e.g. Controller's Eviction Policy).
	void free(DeviceRegionHandle handle);

	/// Rebinds an existing, quiescent allocation to a new logical size without
	/// freeing or allocating device memory. Succeeds only when the allocation's
	/// physical reservation can contain `bytes`; the opaque handle is retained.
	/// RegionManager uses this after an eviction write-back for near-fit reuse.
	bool tryReuse(DeviceRegionHandle handle, std::size_t bytes,
			MemoryKind kind = MemoryKind::Data);

	/// Convenience wrapper (== one enqueueCopyFromHost() + an immediate
	/// flush()): copies `bytes` from `host_src` into `handle`'s allocation at
	/// `dst_offset`. `dst_offset` defaults to 0; Controller::make() passes a
	/// nonzero offset to skip past a Region's prepended dirty-bitmap header
	/// (see gpu/dirty_header.hpp).
	void copyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
										 std::size_t dst_offset = 0);

	/// Convenience wrapper: copies `bytes` starting at `src_offset` bytes
	/// into the allocation backing `handle` out to host memory at `host_dst`,
	/// synchronously. See copyFromHost().
	void copyToHost(DeviceRegionHandle handle, void* host_dst, std::size_t bytes, std::size_t src_offset = 0);

	/// The async, batchable building block copyToHost() is built from (==
	/// one enqueueCopyToHost() + an immediate flush()): enqueues the D2H
	/// copy on DeviceContext's canonical stream without waiting for it to
	/// land. Appends the acquired Lease to `pending`, which the caller must
	/// keep alive -- and must not read `host_dst` through, nor let any
	/// involved handle be freed/compacted -- until flush() returns.
	///
	/// Lets a caller batch many regions' D2H copies onto one stream and pay
	/// for one cudaStreamSynchronize instead of one per region -- see
	/// Controller::writeBackDirtyRegions()'s two-phase use of this for
	/// evictAnchor(). Throws std::out_of_range if `src_offset + bytes`
	/// exceeds `handle`'s allocation.
	void enqueueCopyToHost(DeviceRegionHandle handle, void* host_dst, std::size_t bytes, std::size_t src_offset,
												 std::vector<Lease>& pending);

	/// Mirror image of enqueueCopyToHost() for host-to-device copies, under
	/// the same `pending`/flush() lifetime contract (`host_src` must also
	/// stay valid until flush() returns). See Controller::promoteAnchor()'s
	/// use of this to batch an Anchor's whole footprint into one flush
	/// rather than one make() call at a time. Throws std::out_of_range if
	/// `dst_offset + bytes` exceeds `handle`'s allocation.
	void enqueueCopyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
														std::size_t dst_offset, std::vector<Lease>& pending);
	void enqueueCopyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
														std::size_t dst_offset, TransferBatch& pending);
	void finishTransfers(TransferBatch& pending);

	/// Blocks until every operation previously enqueued on DeviceContext's
	/// canonical stream (via either enqueue*() method or otherwise) has
	/// actually completed. Call exactly once after enqueueing a batch,
	/// before touching destinations or releasing the accumulated Leases.
	void flush();

	/// Total bytes currently outstanding across all live allocations of both
	/// kinds.
	std::size_t bytesAllocated() const;

	/// Bytes currently outstanding for just `kind`.
	std::size_t bytesAllocated(MemoryKind kind) const;

	/// Physical bytes held by live allocations. Equal to bytesAllocated()
	/// under Async, but includes UnitPoolArena rounding under Pooled.
	std::size_t bytesReserved() const;
	std::size_t bytesReserved(MemoryKind kind) const;

	/// Physical bytes one logical `bytes` request reserves from `kind`'s
	/// budget. Under Pooled this rounds up to the configured arena unit; under
	/// Normal it is exactly `bytes`. Cost-aware residency policies use this
	/// instead of underestimating a small Region's actual footprint.
	std::size_t reservationBytes(std::size_t bytes, MemoryKind kind = MemoryKind::Data) const;

	/// Policy-facing snapshots of this pool's configured budget and accounting
	/// granularity. A 1-byte unit under Normal means there is no Arachne-owned
	/// suballocation rounding in that mode.
	std::size_t budgetBytes(MemoryKind kind = MemoryKind::Data) const;
	std::size_t allocationUnitBytes(MemoryKind kind = MemoryKind::Data) const;

	struct CompactionResult {
		std::size_t relocated_count = 0;
		std::size_t bytes_relocated = 0;
	};

	/// Tries to make `kind`'s arena able to satisfy a future
	/// allocateBestFit()-style request of `required_bytes`, by relocating
	/// whichever unpinned live allocations the injected CompactionPolicy
	/// decides are worth moving (see the class overview above). No-op under
	/// Normal -- there's no shared arena to consolidate.
	///
	/// Since allocate()/tryAllocate() already self-heal this way internally,
	/// calling compact() directly is mainly useful for isolated
	/// benchmarking, or for RegionManager's capacity-retry loop, which calls
	/// this *after* flushing any Leases still open from earlier in the same
	/// promotion batch (see RegionManager::allocateWithCompaction()) --
	/// unpinning blocks the original tryAllocate()'s internal retry wasn't
	/// allowed to touch yet.
	///
	/// Holds DeviceRegionPool's own lock for the entire call, so other
	/// allocate()/acquire()/free() calls block until it finishes -- but a
	/// bounded, targeted relocation is typically far cheaper than an older
	/// unconditional full-pool relocation would have been.
	CompactionResult compact(MemoryKind kind, std::size_t required_bytes);

 private:
	struct Allocation : DeviceMemoryBlock {
		// Outstanding-Lease bookkeeping (see Lease's doc comment). One event
		// per distinct stream a Lease has released on -- re-recorded, not
		// re-created, each time, since a stream's own ordering means the
		// latest recording already implies every earlier one on it has passed.
		std::size_t in_use_count = 0;
		std::unordered_map<cudaStream_t, cudaEvent_t> last_used_events;
	};

	UnitPoolArena& arenaFor(MemoryKind kind);
	Allocation allocationFor(DeviceRegionHandle handle);

	// Requires mutex_ already held via `lock` -- see the class overview
	// above for the full snapshot/plan/execute algorithm this implements.
	CompactionResult tryOpenContiguousExtentLocked(MemoryKind kind, std::uint64_t required_units,
																									std::unique_lock<std::mutex>& lock);

	// Called by Lease's destructor.
	void release(DeviceRegionHandle handle, cudaStream_t stream);

	// Shared by free() and compact(): blocks (via cv_, releasing `lock`
	// while waiting) until `id`'s in_use_count is 0, then enqueues a
	// cudaStreamWaitEvent for every event recorded against it so whatever
	// the caller enqueues next is ordered after every prior Lease's work.
	// No-op if `id` is no longer present. Requires mutex_ already held.
	void awaitQuiescentLocked(std::uint64_t id, std::unique_lock<std::mutex>& lock);

	DeviceContext& device_;
	std::unique_ptr<DeviceMemoryManager> memory_manager_;
	PinnedHostPool pinned_host_pool_;
	std::unique_ptr<CompactionPolicy> compaction_policy_;
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::unordered_map<std::uint64_t, Allocation> allocations_;
	std::uint64_t next_id_ = 1;
};

}  // namespace arachne::gpu
