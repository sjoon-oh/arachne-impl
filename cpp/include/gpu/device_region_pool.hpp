#pragma once

#include <cstddef>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <unordered_map>

#include <cuda/memory_resource>
#include <cuda_runtime.h>

#include "gpu/device_context.hpp"
#include "gpu/device_region_handle.hpp"

namespace arachne::gpu {

/// Arachne-owned GPU memory allocator backing every Region (see
/// core/region_manager.hpp). Per the Anchor-centric residency policy:
/// promote/evict decisions are made per-Region based on the depending
/// Anchors' observed hotness/latency/transfer-cost, not on fixed-size
/// address-based pages -- so allocations here are variable-sized and
/// identified by opaque handle, not laid out on any fixed slab grid.
/// Adapters never call cudaMalloc directly for memory a Region is meant to
/// cover; they go through the DeviceRegionPool threaded down from Controller so
/// Arachne, not the index, accounts for (and eventually migrates)
/// residency.
///
/// Backed by DeviceContext's two resources (dataResource()/
/// metadataResource(), routed to by `kind` -- see MemoryKind), each built
/// according to DeviceContext::allocationPolicy() (see AllocationPolicy).
/// DeviceRegionPool itself never checks which policy is active -- both
/// alternatives present the identical cuda::mr::any_resource
/// allocate()/deallocate() surface, which is the whole point of that
/// type erasure: swapping policies is a DeviceContext-construction-time
/// choice, invisible here.
///
/// Thread-safe: every method takes DeviceRegionPool's own lock, mirroring
/// RegionManager's own concurrency contract (Controller is called
/// concurrently the same way RegionManager is).
class DeviceRegionPool {
 public:
	/// RAII marker: "the current holder is actively using `handle`'s device
	/// memory, work against it may still be in flight on stream()". While a
	/// Lease is alive, DeviceRegionPool guarantees the handle it covers will not be
	/// relocated (compact()) or reclaimed (free(), including via Eviction --
	/// see Controller) out from under it -- both wait for every outstanding
	/// Lease on a handle to be released before touching its memory (see
	/// acquire() and the shared internal mechanism both compact() and free()
	/// use).
	///
	/// Releasing a Lease (its destructor) does not itself prove the GPU has
	/// finished any work enqueued against stream() while the Lease was held
	/// -- work enqueued asynchronously can still be running after the host
	/// call that launched it returns. So release() records a CUDA event on
	/// stream() marking "everything enqueued on this stream up to here, by
	/// this Lease"; a future compact()/free() call waits on that event (via
	/// cudaStreamWaitEvent, not a host-blocking sync) before reusing the
	/// memory, so no host-side stall is imposed on the common case where
	/// nothing needs to wait.
	///
	/// Move-only: ownership of "currently in use" is unique per Lease.
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

	explicit DeviceRegionPool(DeviceContext& device);
	~DeviceRegionPool();

	DeviceRegionPool(const DeviceRegionPool&) = delete;
	DeviceRegionPool& operator=(const DeviceRegionPool&) = delete;

	/// Allocates `bytes` of device memory from the resource matching `kind`
	/// (see MemoryKind) and returns a handle to it. Throws (propagated from
	/// the underlying cuda::mr::any_resource) if the allocation actually
	/// fails -- callers that need a non-throwing, capacity-aware attempt
	/// (Controller's Promotion path) should use tryAllocate() below instead.
	DeviceRegionHandle allocate(std::size_t bytes, MemoryKind kind = MemoryKind::Data);

	/// True if allocating `bytes` more of `kind` would stay within
	/// DeviceContext::budgetBytes(kind) -- Arachne's own self-imposed ceiling
	/// on how much `kind` memory it is willing to have resident, checked
	/// against what's currently outstanding (bytesAllocated(kind)). A
	/// snapshot, not a reservation: nothing stops a concurrent
	/// allocate()/tryAllocate() from being accepted in between a caller
	/// checking this and acting on it (see tryAllocate()'s own doc comment
	/// for why that race is acceptable here).
	bool hasCapacity(std::size_t bytes, MemoryKind kind = MemoryKind::Data) const;

	/// Capacity-aware allocate(): returns std::nullopt instead of throwing,
	/// both when hasCapacity() already says no (the common, cheap case --
	/// avoids an allocation attempt Arachne itself has decided not to permit)
	/// and when the underlying resource's allocate() throws despite
	/// hasCapacity() saying yes (e.g. actual GPU memory pressure under
	/// AllocationPolicy::Naive, which never pre-reserves -- see
	/// AllocationPolicy's doc comment). This is what lets Controller::make()
	/// treat "no room right now" as an ordinary, retryable outcome (evict a
	/// victim, try again) instead of an exception to propagate -- see its own
	/// doc comment.
	///
	/// The race hasCapacity() alone leaves open (two concurrent callers both
	/// observing enough headroom for their own request, then both actually
	/// allocating and together exceeding budget by a bounded amount) is
	/// accepted rather than closed with a reservation step: Controller only
	/// ever calls this while holding intent to immediately use the result,
	/// and a bounded, occasional overshoot of a self-imposed budget is far
	/// cheaper to tolerate than serializing every allocate() behind a second
	/// lock acquisition across this call and the real one.
	std::optional<DeviceRegionHandle> tryAllocate(std::size_t bytes, MemoryKind kind = MemoryKind::Data);

	/// Marks `handle` in use on `stream` until the returned Lease is
	/// destroyed -- see Lease's own doc comment for what that guarantees.
	/// This is the *only* way to resolve a handle to its device pointer:
	/// there used to be a bare access() that returned a raw pointer with no
	/// such guarantee, but every caller that resolved a pointer without
	/// holding a Lease around its actual use was exposed to compact()
	/// relocating (or free() reclaiming) the memory out from under it --
	/// including, subtly, this class's own copyFromHost()/copyToHost() below
	/// before they were changed to acquire() internally. Throws
	/// std::invalid_argument if `handle` was never returned by allocate() on
	/// this pool, or was already freed.
	Lease acquire(DeviceRegionHandle handle, cudaStream_t stream);

	/// Same as the two-argument overload, on DeviceContext's own canonical
	/// stream.
	Lease acquire(DeviceRegionHandle handle);

	/// Releases the allocation backing `handle`. No-op if `handle` is
	/// invalid or was already freed. Blocks until every outstanding Lease on
	/// `handle` has been released and the GPU has actually caught up to the
	/// point each one was released at (see Lease's doc comment) -- callers
	/// (e.g. Controller executing an Eviction Policy decision) should expect
	/// this to wait rather than fail or skip: an eviction target was already
	/// chosen for a reason, so waiting for it to become safe to reclaim is
	/// the correct behavior, not skipping it.
	void free(DeviceRegionHandle handle);

	/// Convenience wrapper: copies `bytes` from host memory at `host_src`
	/// into the allocation backing `handle` starting at `dst_offset` bytes
	/// into it, synchronously (== one enqueueCopyFromHost() + an immediate
	/// flush()) -- provided so callers don't need direct access to
	/// DeviceContext's stream, or their own Lease, just to move bytes in.
	/// `dst_offset` defaults to 0; Controller::make() passes a nonzero offset
	/// to copy a Region's data past its prepended dirty-bitmap header (see
	/// gpu/dirty_header.hpp).
	void copyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
										 std::size_t dst_offset = 0);

	/// Convenience wrapper: copies `bytes` starting at `src_offset` bytes
	/// into the allocation backing `handle` out to host memory at `host_dst`,
	/// synchronously. See copyFromHost().
	void copyToHost(DeviceRegionHandle handle, void* host_dst, std::size_t bytes, std::size_t src_offset = 0);

	/// The async, batchable building block copyToHost() is built from
	/// (copyToHost() == one enqueueCopyToHost() + an immediate flush()):
	/// enqueues a device-to-host copy of `bytes` starting at `src_offset`
	/// bytes into the allocation backing `handle`, into host memory at
	/// `host_dst`, on DeviceContext's canonical stream -- but does *not*
	/// wait for it to land. Appends the Lease it acquires (guarding against
	/// a concurrent compact()/free() the same way copyToHost()'s internal
	/// one does) to `pending`, which the caller must keep alive -- and must
	/// not read `host_dst` through, nor let any handle involved be
	/// freed/compacted -- until flush() has been called and returned; only
	/// then is it safe to drop `pending` (releasing its Leases) or read the
	/// copied-into host memory.
	///
	/// This is what lets a caller batch many regions' device-to-host copies
	/// onto one stream and pay for exactly one cudaStreamSynchronize instead
	/// of one per region -- see Controller::writeBackDirtyRegions()'s
	/// two-phase (gather headers, then gather dirty payloads) use of this
	/// for evictAnchor(). Same bounds check as copyToHost(): throws
	/// std::out_of_range if `src_offset + bytes` exceeds `handle`'s
	/// allocation.
	void enqueueCopyToHost(DeviceRegionHandle handle, void* host_dst, std::size_t bytes, std::size_t src_offset,
												 std::vector<Lease>& pending);

	/// Mirror image of enqueueCopyToHost(): enqueues a host-to-device copy of
	/// `bytes` from host memory at `host_src` into the allocation backing
	/// `handle`, starting at `dst_offset` bytes into it, without waiting for
	/// it to land. Appends the acquired Lease to `pending`, under the exact
	/// same lifetime contract as enqueueCopyToHost() -- `host_src` must also
	/// stay valid until flush() returns. Lets a caller batch many regions'
	/// host-to-device copies onto one stream and pay for exactly one
	/// cudaStreamSynchronize instead of one per region -- see
	/// Controller::promoteAnchor()'s use of this to copy every Region in an
	/// Anchor's footprint in one batch rather than one make() call at a time.
	/// Same bounds check as copyFromHost(): throws std::out_of_range if
	/// `dst_offset + bytes` exceeds `handle`'s allocation.
	void enqueueCopyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
														std::size_t dst_offset, std::vector<Lease>& pending);

	/// Blocks until every operation previously enqueued on DeviceContext's
	/// canonical stream (via enqueueCopyToHost()/enqueueCopyFromHost() or
	/// otherwise) has actually completed. Callers using either enqueue*()
	/// method must call this exactly once after enqueueing everything in a
	/// batch, before touching any of the destinations or releasing the
	/// Leases they accumulated.
	void flush();

	/// Total bytes currently outstanding across all live allocations of both
	/// kinds -- the sizing signal the eventual Anchor-driven promote/evict
	/// policy needs to reason about a GPU memory budget (that policy only
	/// ever concerns MemoryKind::Data, hence the per-kind overload below).
	std::size_t bytesAllocated() const;

	/// Bytes currently outstanding for just `kind`.
	std::size_t bytesAllocated(MemoryKind kind) const;

	struct CompactionResult {
		std::size_t relocated_count = 0;
		std::size_t bytes_relocated = 0;
	};

	/// Relocates every currently-live allocation of `kind` through a fresh
	/// allocate() + cudaMemcpyDeviceToDevice() + free() cycle, giving the
	/// underlying resource a chance to re-coalesce more tightly. Adapted
	/// from DynaSOAr's (github.com/prg-titech/dynasoar) parallel_defrag,
	/// which scans object *blocks* for low occupancy and only relocates
	/// objects out of sparse ones -- we don't have DynaSOAr's block/bitmap
	/// structure to scan (our allocations are opaque variable-sized byte
	/// ranges, not statically-typed objects in fixed-size blocks; our
	/// resources are further type-erased behind cuda::mr::any_resource, so
	/// there's no free-list to inspect even if we did), so this relocates
	/// everything of `kind` unconditionally rather than picking sparse
	/// candidates out of a structure we can't see into.
	///
	/// Deliberately has no opinion on *when* it should run -- that decision
	/// belongs to the caller (Controller), not here. In particular this is
	/// not a periodic/occupancy-triggered background job: Arachne's pool is
	/// meant to run near-full most of the time (that's the point of
	/// promoting aggressively), so an "occupancy is low" trigger would
	/// rarely fire exactly when it's actually needed. The intended trigger
	/// is allocation failure in the Eviction -> Compaction -> Promotion
	/// pipeline: Controller evicts cold Regions, attempts the promotion's
	/// allocate(), and only calls this -- then retries -- if that allocate()
	/// fails despite there being enough aggregate free bytes (i.e.
	/// fragmentation, not an actual out-of-memory condition).
	///
	/// A no-op under AllocationPolicy::Naive: every Region there already has
	/// its own independent cudaMalloc'd block (no shared arena), so there is
	/// nothing to consolidate -- see AllocationPolicy's doc comment.
	///
	/// Safety: for each live allocation of `kind`, waits for every
	/// outstanding Lease on it to be released and the GPU to actually reach
	/// each release's recorded point (the same mechanism free() uses -- see
	/// Lease's doc comment) before copying it, so the device-to-device copy
	/// never races an in-flight kernel/copy against the old address.
	///
	/// Cost: while running, both the old and new copy of everything being
	/// relocated are live simultaneously, so `kind`'s resource needs
	/// roughly double the current live bytes of `kind` free to complete.
	///
	/// Concurrency: holds DeviceRegionPool's own lock for the entire call
	/// (including waiting out any outstanding Leases, the device-to-device
	/// copies, and the stream sync they wait on), so other
	/// allocate()/acquire()/free() calls on this DeviceRegionPool block
	/// until it finishes -- simpler and safer than interleaving, at the cost
	/// of pausing everything else while compaction runs; revisit with a
	/// snapshot/no-lock-move/reconcile split (the same shape ASRoutingCache's
	/// own compaction already uses) if that pause becomes a real problem.
	CompactionResult compact(MemoryKind kind);

 private:
	struct Allocation {
		void* device_ptr = nullptr;
		std::size_t bytes = 0;
		MemoryKind kind = MemoryKind::Data;

		// Outstanding-Lease bookkeeping (see Lease's doc comment). One event
		// per distinct stream a Lease has ever released on for this
		// allocation -- re-recorded (not re-created) each time that same
		// stream is used again, since a stream's own ordering means the
		// latest recording already implies every earlier one on it has
		// passed.
		std::size_t in_use_count = 0;
		std::unordered_map<cudaStream_t, cudaEvent_t> last_used_events;
	};

	cuda::mr::any_resource<cuda::mr::device_accessible>& resourceFor(MemoryKind kind);
	Allocation allocationFor(DeviceRegionHandle handle);

	// Called by Lease's destructor.
	void release(DeviceRegionHandle handle, cudaStream_t stream);

	// Shared by free() and compact(): blocks (via cv_, releasing `lock`
	// while waiting) until `id`'s in_use_count is 0, then enqueues a
	// cudaStreamWaitEvent on DeviceContext's canonical stream for every
	// event recorded against it and destroys/clears them -- after this
	// returns, anything subsequently enqueued on that same stream (the
	// actual deallocate()/cudaMemcpyAsync() call the caller makes next) is
	// correctly ordered after every prior Lease's work. No-op if `id` is no
	// longer present in allocations_. Requires mutex_ already held via
	// `lock`.
	void awaitQuiescentLocked(std::uint64_t id, std::unique_lock<std::mutex>& lock);

	DeviceContext& device_;
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::unordered_map<std::uint64_t, Allocation> allocations_;
	std::uint64_t next_id_ = 1;
};

}  // namespace arachne::gpu
