#pragma once

#include <cstddef>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

#include <cuda/memory_resource>
#include <cuda_runtime.h>

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
/// Backed by DeviceContext's two resources (dataResource()/
/// metadataResource(), routed to by `kind` -- see MemoryKind), each built
/// according to DeviceContext::allocationPolicy() (see AllocationPolicy).
/// StitchPool itself never checks which policy is active -- both
/// alternatives present the identical cuda::mr::any_resource
/// allocate()/deallocate() surface, which is the whole point of that
/// type erasure: swapping policies is a DeviceContext-construction-time
/// choice, invisible here.
///
/// Thread-safe: every method takes StitchPool's own lock, mirroring
/// AnchorManager's own concurrency contract (Controller is called
/// concurrently the same way AnchorManager is).
class StitchPool {
 public:
	/// RAII marker: "the current holder is actively using `handle`'s device
	/// memory, work against it may still be in flight on stream()". While a
	/// Lease is alive, StitchPool guarantees the handle it covers will not be
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

		/// The handle's device pointer, resolved at acquire() time. Stable for
		/// this Lease's lifetime (nothing relocates/frees a handle with an
		/// outstanding Lease) -- unlike access()'s pointer, this one is safe to
		/// hold onto and pass into an asynchronous kernel launch on stream().
		void* ptr() const { return ptr_; }

		cudaStream_t stream() const { return stream_; }

	 private:
		friend class StitchPool;
		Lease(StitchPool& pool, StitchHandle handle, cudaStream_t stream, void* ptr);

		StitchPool* pool_;  // null after being moved from
		StitchHandle handle_;
		cudaStream_t stream_;
		void* ptr_;
	};

	explicit StitchPool(DeviceContext& device);
	~StitchPool();

	StitchPool(const StitchPool&) = delete;
	StitchPool& operator=(const StitchPool&) = delete;

	/// Allocates `bytes` of device memory from the resource matching `kind`
	/// (see MemoryKind) and returns a handle to it.
	StitchHandle allocate(std::size_t bytes, MemoryKind kind = MemoryKind::Data);

	/// Resolves `handle` to its device pointer. Throws std::invalid_argument
	/// if `handle` was never returned by allocate() on this pool, or was
	/// already freed. The returned pointer is only valid until the next
	/// compact() call relocates it -- fine for callers that resolve and
	/// finish using it synchronously (e.g. copyFromHost()/copyToHost() below,
	/// which sync DeviceContext's stream before returning), but NOT safe to
	/// hold onto across anything that launches asynchronous GPU work and
	/// returns before it finishes -- use acquire() for that instead.
	void* access(StitchHandle handle, AccessMode mode);

	/// Marks `handle` in use on `stream` until the returned Lease is
	/// destroyed -- see Lease's own doc comment for what that guarantees and
	/// why access() alone isn't enough for asynchronous GPU work. Throws
	/// std::invalid_argument under the same conditions as access().
	Lease acquire(StitchHandle handle, cudaStream_t stream);

	/// Same as the two-argument overload, on DeviceContext's own canonical
	/// stream.
	Lease acquire(StitchHandle handle);

	/// Releases the allocation backing `handle`. No-op if `handle` is
	/// invalid or was already freed. Blocks until every outstanding Lease on
	/// `handle` has been released and the GPU has actually caught up to the
	/// point each one was released at (see Lease's doc comment) -- callers
	/// (e.g. Controller executing an Eviction Policy decision) should expect
	/// this to wait rather than fail or skip: an eviction target was already
	/// chosen for a reason, so waiting for it to become safe to reclaim is
	/// the correct behavior, not skipping it.
	void free(StitchHandle handle);

	/// Convenience wrapper: copies `bytes` from host memory at `host_src`
	/// into the allocation backing `handle`, synchronously (blocks until the
	/// copy completes, on DeviceContext's own stream). Equivalent to
	/// cudaMemcpy(access(handle, Write), host_src, bytes,
	/// cudaMemcpyHostToDevice) plus a stream sync; provided so callers don't
	/// need direct access to DeviceContext's stream just to move bytes in.
	void copyFromHost(StitchHandle handle, const void* host_src, std::size_t bytes);

	/// Convenience wrapper: copies `bytes` out of the allocation backing
	/// `handle` into host memory at `host_dst`, synchronously. See
	/// copyFromHost().
	void copyToHost(StitchHandle handle, void* host_dst, std::size_t bytes);

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
	/// pipeline: Controller evicts cold Stitches, attempts the promotion's
	/// allocate(), and only calls this -- then retries -- if that allocate()
	/// fails despite there being enough aggregate free bytes (i.e.
	/// fragmentation, not an actual out-of-memory condition).
	///
	/// A no-op under AllocationPolicy::Naive: every Stitch there already has
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
	/// Concurrency: holds StitchPool's own lock for the entire call
	/// (including waiting out any outstanding Leases, the device-to-device
	/// copies, and the stream sync they wait on), so other
	/// allocate()/access()/acquire()/free() calls on this StitchPool block
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
	Allocation allocationFor(StitchHandle handle);

	// Called by Lease's destructor.
	void release(StitchHandle handle, cudaStream_t stream);

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
