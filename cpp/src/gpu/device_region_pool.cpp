#include "gpu/device_region_pool.hpp"

#include <cuda_runtime.h>
#include <rmm/aligned.hpp>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "logging.hpp"
#include "telemetry/trace.hpp"

namespace arachne::gpu {

namespace {

void CheckCuda(cudaError_t status, const char* what) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
	}
}

std::uint64_t RequiredUnits(std::size_t bytes, std::size_t unit_bytes) {
	return (static_cast<std::uint64_t>(bytes) + unit_bytes - 1) / unit_bytes;
}

}  // namespace

// ---------------------------------------------------------------------------
// Lease
// ---------------------------------------------------------------------------

DeviceRegionPool::Lease::Lease(DeviceRegionPool& pool, DeviceRegionHandle handle, cudaStream_t stream, void* ptr)
		: pool_(&pool), handle_(handle), stream_(stream), ptr_(ptr) {}

DeviceRegionPool::Lease::Lease(Lease&& other) noexcept
		: pool_(other.pool_), handle_(other.handle_), stream_(other.stream_), ptr_(other.ptr_) {
	other.pool_ = nullptr;
}

DeviceRegionPool::Lease& DeviceRegionPool::Lease::operator=(Lease&& other) noexcept {
	if (this == &other) return *this;
	if (pool_ != nullptr) pool_->release(handle_, stream_);
	pool_ = other.pool_;
	handle_ = other.handle_;
	stream_ = other.stream_;
	ptr_ = other.ptr_;
	other.pool_ = nullptr;
	return *this;
}

DeviceRegionPool::Lease::~Lease() {
	if (pool_ != nullptr) pool_->release(handle_, stream_);
}

// ---------------------------------------------------------------------------
// DeviceRegionPool
// ---------------------------------------------------------------------------

DeviceRegionPool::DeviceRegionPool(DeviceContext& device, std::unique_ptr<CompactionPolicy> compaction_policy)
		: device_(device),
			memory_manager_(MakeDeviceMemoryManager(device)),
			compaction_policy_(compaction_policy != nullptr ? std::move(compaction_policy)
																												: std::make_unique<TargetedCompactionPolicy>()) {}

// Safety net, not the expected path: every Region should already have had
// its device residency reclaimed via RegionManager::forget()/
// removeDependency() before the pool backing it goes away. Reclaims
// whatever is still outstanding rather than leaking it: event objects
// always, and -- only under Normal, where each allocation owns its own
// cudaMalloc'd block -- the device memory itself. Under Pooled there is
// nothing more to do per-allocation: the arena's one big buffer is reclaimed
// later, wholesale, by DeviceContext's own UnitPoolArena destructor.
DeviceRegionPool::~DeviceRegionPool() {
	for (auto& [id, allocation] : allocations_) {
		for (auto& [stream, event] : allocation.last_used_events) {
			cudaEventDestroy(event);
		}
		memory_manager_->deallocate(allocation);
	}
	device_.resources().sync_stream();
}

UnitPoolArena& DeviceRegionPool::arenaFor(MemoryKind kind) {
	// Always non-null here: every call site is already gated on
	// device_.allocationPolicy() == Pooled, which is exactly when
	// DeviceContext constructs both arenas.
	UnitPoolArena* arena = memory_manager_->arena(kind);
	if (arena == nullptr) throw std::logic_error("DeviceRegionPool: backend has no arena");
	return *arena;
}

DeviceRegionPool::Allocation DeviceRegionPool::allocationFor(DeviceRegionHandle handle) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allocations_.find(handle.id);
	if (it == allocations_.end()) {
		throw std::invalid_argument("DeviceRegionPool: handle is not a live allocation");
	}
	return it->second;
}

void DeviceRegionPool::awaitQuiescentLocked(std::uint64_t id, std::unique_lock<std::mutex>& lock) {
	cv_.wait(lock, [&] {
		auto it = allocations_.find(id);
		return it == allocations_.end() || it->second.in_use_count == 0;
	});

	auto it = allocations_.find(id);
	if (it == allocations_.end()) return;  // freed by someone else while we waited

	// Enqueue a wait for each recorded event on our own canonical stream so
	// whatever we enqueue next (deallocate/relocation copy/memcpy) is ordered
	// after all of them, without host-blocking -- the events are then unneeded.
	for (auto& [stream, event] : it->second.last_used_events) {
		cudaStreamWaitEvent(device_.resources().get_stream().value(), event, 0);
		cudaEventDestroy(event);
	}
	it->second.last_used_events.clear();
}

void DeviceRegionPool::release(DeviceRegionHandle handle, cudaStream_t stream) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allocations_.find(handle.id);
	if (it == allocations_.end()) return;  // freed out from under an outstanding Lease shouldn't
																					// happen (free() waits for in_use_count == 0
																					// first), but tolerate it defensively.

	Allocation& allocation = it->second;
	if (allocation.in_use_count > 0) --allocation.in_use_count;

	auto event_it = allocation.last_used_events.find(stream);
	cudaEvent_t event;
	if (event_it == allocation.last_used_events.end()) {
		cudaEventCreate(&event);
		allocation.last_used_events.emplace(stream, event);
	} else {
		event = event_it->second;
	}
	cudaEventRecord(event, stream);

	cv_.notify_all();
}

DeviceRegionHandle DeviceRegionPool::allocate(std::size_t bytes, MemoryKind kind) {
	std::unique_lock<std::mutex> lock(mutex_);
	DeviceMemoryBlock block;
	try {
		block = memory_manager_->allocate(bytes, kind);
	} catch (...) {
		if (memory_manager_->arena(kind) == nullptr) throw;
		tryOpenContiguousExtentLocked(kind, RequiredUnits(bytes, arenaFor(kind).unitBytes()), lock);
		block = memory_manager_->allocate(bytes, kind);
	}
	std::uint64_t id = next_id_++;
	Allocation allocation;
	static_cast<DeviceMemoryBlock&>(allocation) = std::move(block);
	allocations_.emplace(id, std::move(allocation));
	return DeviceRegionHandle{id};
}

bool DeviceRegionPool::hasCapacity(std::size_t bytes, MemoryKind kind) const {
	return bytesAllocated(kind) + bytes <= device_.budgetBytes(kind);
}

std::optional<DeviceRegionHandle> DeviceRegionPool::tryAllocate(std::size_t bytes, MemoryKind kind) {
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "tryAllocate");
	if (!hasCapacity(bytes, kind)) {
		ARACHNE_LOG_DEBUG("DeviceRegionPool::tryAllocate: {} bytes of kind {} would exceed budget ({} already allocated)",
											 bytes, static_cast<int>(kind), bytesAllocated(kind));
		return std::nullopt;
	}
	try {
		return allocate(bytes, kind);
	} catch (const std::exception& e) {
		ARACHNE_LOG_DEBUG("DeviceRegionPool::tryAllocate: allocate({} bytes) failed despite passing the budget check: {}",
											 bytes, e.what());
		return std::nullopt;
	}
}

// Cross-stream hand-off: a Lease's release() on one stream only records an
// event, it doesn't wait for the GPU to catch up (see Lease's doc comment)
// -- so before handing out a Lease on `stream`, acquire() must make
// `stream`'s upcoming work wait on every *other* stream's still-pending
// release event first, via a GPU-side cudaStreamWaitEvent (never a host
// block):
//
//   stream A: ...kernel...--release()-->[event A]
//   stream B: acquire() --cudaStreamWaitEvent(B,eventA)--> ...kernel...
//                          (event A now consumed: destroyed + erased)
//
// This is what lets independent streams (per-worker compute streams, the
// management stream) safely interleave access to the same allocation --
// without it they'd share no ordering at all, since they no longer share
// one canonical stream's implicit FIFO order. Consuming each waited-on event
// is safe because `stream`'s own future release() event will already be
// ordered-after everything that entry represented, so a later acquire() on
// some third stream only needs to wait on `stream`'s eventual release, not
// re-wait on this one too.
DeviceRegionPool::Lease DeviceRegionPool::acquire(DeviceRegionHandle handle, cudaStream_t stream) {
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "acquire");
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allocations_.find(handle.id);
	if (it == allocations_.end()) {
		throw std::invalid_argument("DeviceRegionPool::acquire: handle is not a live allocation");
	}

	// Wait on every *other* stream's pending event (see overview above);
	// entries already on `stream` are left alone -- same-stream ordering is
	// already implicit in CUDA's own enqueue order.
	for (auto event_it = it->second.last_used_events.begin(); event_it != it->second.last_used_events.end();) {
		if (event_it->first == stream) {
			++event_it;
			continue;
		}
		cudaStreamWaitEvent(stream, event_it->second, 0);
		cudaEventDestroy(event_it->second);
		event_it = it->second.last_used_events.erase(event_it);
	}

	++it->second.in_use_count;
	return Lease(*this, handle, stream, it->second.device_ptr);
}

DeviceRegionPool::Lease DeviceRegionPool::acquire(DeviceRegionHandle handle) {
	return acquire(handle, device_.resources().get_stream().value());
}

void DeviceRegionPool::free(DeviceRegionHandle handle) {
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "free");
	Allocation allocation;
	{
		std::unique_lock<std::mutex> lock(mutex_);
		if (allocations_.find(handle.id) == allocations_.end()) return;

		awaitQuiescentLocked(handle.id, lock);  // may block until every Lease is released

		auto it = allocations_.find(handle.id);
		if (it == allocations_.end()) return;  // freed by someone else while we waited
		allocation = it->second;
		allocations_.erase(it);
	}
	memory_manager_->deallocate(allocation);
}

bool DeviceRegionPool::tryReuse(DeviceRegionHandle handle, std::size_t bytes, MemoryKind kind) {
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "tryReuse");
	std::unique_lock<std::mutex> lock(mutex_);
	auto it = allocations_.find(handle.id);
	if (it == allocations_.end() || it->second.kind != kind) return false;
	if (memory_manager_->reservationBytes(bytes, kind) > it->second.reserved_bytes) return false;

	awaitQuiescentLocked(handle.id, lock);
	it = allocations_.find(handle.id);
	if (it == allocations_.end() || it->second.kind != kind) return false;
	it->second.bytes = bytes;
	return true;
}

void DeviceRegionPool::copyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
																		 std::size_t dst_offset) {
	TransferBatch pending;
	enqueueCopyFromHost(handle, host_src, bytes, dst_offset, pending);
	finishTransfers(pending);
	// `pending`'s Lease releases here -- safe, flush() already proved the
	// copy landed.
}

void DeviceRegionPool::enqueueCopyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
																						std::size_t dst_offset, std::vector<Lease>& pending) {
	if (dst_offset + bytes > allocationFor(handle).bytes) {
		throw std::out_of_range("DeviceRegionPool::enqueueCopyFromHost: dst_offset + bytes exceeds the allocation's size");
	}
	Lease lease = acquire(handle);  // guards the copy against a concurrent compact()/free()
	void* dst = static_cast<std::byte*>(lease.ptr()) + dst_offset;
	CheckCuda(cudaMemcpyAsync(dst, host_src, bytes, cudaMemcpyHostToDevice, lease.stream()),
						"DeviceRegionPool::enqueueCopyFromHost: cudaMemcpyAsync");
	pending.push_back(std::move(lease));
}

void DeviceRegionPool::enqueueCopyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
		std::size_t dst_offset, TransferBatch& pending) {
	if (dst_offset + bytes > allocationFor(handle).bytes) {
		throw std::out_of_range("DeviceRegionPool::enqueueCopyFromHost: copy exceeds allocation");
	}
	PinnedHostPool::Buffer staging = pinned_host_pool_.acquire(bytes);
	if (bytes > 0) std::memcpy(staging.data(), host_src, bytes);
	Lease lease = acquire(handle);
	void* dst = static_cast<std::byte*>(lease.ptr()) + dst_offset;
	CheckCuda(cudaMemcpyAsync(dst, staging.data(), bytes, cudaMemcpyHostToDevice, lease.stream()),
					"DeviceRegionPool::enqueueCopyFromHost(pinned): cudaMemcpyAsync");
	pending.pinned_sources.push_back(std::move(staging));
	pending.leases.push_back(std::move(lease));
}

void DeviceRegionPool::finishTransfers(TransferBatch& pending) {
	if (pending.leases.empty()) return;
	cudaEvent_t ready;
	CheckCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
					"DeviceRegionPool::finishTransfers: cudaEventCreateWithFlags");
	CheckCuda(cudaEventRecord(ready, device_.managementStream()),
					"DeviceRegionPool::finishTransfers: cudaEventRecord");
	CheckCuda(cudaEventSynchronize(ready), "DeviceRegionPool::finishTransfers: cudaEventSynchronize");
	cudaEventDestroy(ready);
	pending.leases.clear();
	pending.pinned_sources.clear();
}

void DeviceRegionPool::copyToHost(DeviceRegionHandle handle, void* host_dst, std::size_t bytes,
																	 std::size_t src_offset) {
	std::vector<Lease> pending;
	enqueueCopyToHost(handle, host_dst, bytes, src_offset, pending);
	flush();
	// `pending`'s Lease releases here -- safe, flush() already proved the
	// copy landed.
}

void DeviceRegionPool::enqueueCopyToHost(DeviceRegionHandle handle, void* host_dst, std::size_t bytes,
																					std::size_t src_offset, std::vector<Lease>& pending) {
	if (src_offset + bytes > allocationFor(handle).bytes) {
		throw std::out_of_range("DeviceRegionPool::enqueueCopyToHost: src_offset + bytes exceeds the allocation's size");
	}
	Lease lease = acquire(handle);  // guards the copy against a concurrent compact()/free()
	const void* src = static_cast<const std::byte*>(lease.ptr()) + src_offset;
	CheckCuda(cudaMemcpyAsync(host_dst, src, bytes, cudaMemcpyDeviceToHost, lease.stream()),
						"DeviceRegionPool::enqueueCopyToHost: cudaMemcpyAsync");
	pending.push_back(std::move(lease));
}

void DeviceRegionPool::flush() {
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "flush");
	device_.resources().sync_stream();
}

std::size_t DeviceRegionPool::bytesAllocated() const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::size_t total = 0;
	for (const auto& [id, allocation] : allocations_) total += allocation.bytes;
	return total;
}

std::size_t DeviceRegionPool::bytesAllocated(MemoryKind kind) const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::size_t total = 0;
	for (const auto& [id, allocation] : allocations_) {
		if (allocation.kind == kind) total += allocation.bytes;
	}
	return total;
}

std::size_t DeviceRegionPool::bytesReserved() const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::size_t total = 0;
	for (const auto& [id, allocation] : allocations_) total += allocation.reserved_bytes;
	return total;
}

std::size_t DeviceRegionPool::bytesReserved(MemoryKind kind) const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::size_t total = 0;
	for (const auto& [id, allocation] : allocations_) {
		if (allocation.kind == kind) total += allocation.reserved_bytes;
	}
	return total;
}

std::size_t DeviceRegionPool::reservationBytes(std::size_t bytes, MemoryKind kind) const {
	return memory_manager_->reservationBytes(bytes, kind);
}

std::size_t DeviceRegionPool::budgetBytes(MemoryKind kind) const { return memory_manager_->budgetBytes(kind); }

std::size_t DeviceRegionPool::allocationUnitBytes(MemoryKind kind) const {
	return memory_manager_->allocationUnitBytes(kind);
}

DeviceRegionPool::CompactionResult DeviceRegionPool::compact(MemoryKind kind, std::size_t required_bytes) {
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "compact");
	if (memory_manager_->arena(kind) == nullptr) {
		// No shared arena under Normal -- see the declaration's doc comment.
		return {};
	}

	std::unique_lock<std::mutex> lock(mutex_);
	UnitPoolArena& arena = arenaFor(kind);
	std::uint64_t required_units = RequiredUnits(required_bytes, arena.unitBytes());
	return tryOpenContiguousExtentLocked(kind, required_units, lock);
}

// The shared snapshot -> plan -> execute pipeline both allocatePooled()'s
// internal self-heal retry and the public compact() are built on -- the
// only difference between those two callers is *when* they invoke this
// (allocatePooled(): reactively, only after its own best-fit already
// failed; compact(): proactively, on explicit caller request):
//
//   allocations_ (kind, in_use_count==0)      arena.freeExtentsByAddress()
//              |                                         |
//              `------------> compaction_policy_->plan() <'
//                                    |
//                              Plan{moves[], feasible}
//                                    |
//              for each move, in order (mutex_ held throughout):
//                re-validate block still present & unpinned  (cheap
//                  insurance -- can't actually fail, nothing has changed
//                  since the snapshot above)
//                awaitQuiescentLocked()  (non-blocking: already unpinned)
//                arena.claim(to) -> arena.relocate(from, to)  (D2D copy)
//                update device_ptr/unit_range -> arena.free(from)
//
// Requires mutex_ already held via `lock`. Returns {} without ever
// consulting compaction_policy_ if `kind`'s largest free extent already
// satisfies required_units (nothing to do) or its totalFreeUnits() doesn't
// (genuine shortfall -- no plan can help).
DeviceRegionPool::CompactionResult DeviceRegionPool::tryOpenContiguousExtentLocked(
		MemoryKind kind, std::uint64_t required_units, std::unique_lock<std::mutex>& lock) {
	// Own trace scope, distinct from "compact"'s parent scope, since this also
	// runs from allocatePooled()'s self-heal retry -- otherwise that path's
	// relocation cost was silently folded into "tryAllocate"'s own duration.
	ARACHNE_TRACE_SCOPE("DeviceRegionPool", "tryOpenContiguousExtentLocked");
	UnitPoolArena& arena = arenaFor(kind);

	if (arena.largestFreeExtent() >= required_units) return {};  // nothing to do
	if (arena.totalFreeUnits() < required_units) return {};      // genuine shortfall -- no plan can help

	std::vector<MovableBlock> movable;
	for (const auto& [id, allocation] : allocations_) {
		if (allocation.kind != kind) continue;
		if (allocation.in_use_count != 0) continue;  // pinned -- never offered, never waited on
		movable.push_back(MovableBlock{id, allocation.unit_range});
	}
	std::sort(movable.begin(), movable.end(), [](const MovableBlock& a, const MovableBlock& b) {
		return a.range.start_unit < b.range.start_unit;
	});

	CompactionPolicy::Plan plan = compaction_policy_->plan(arena, movable, required_units);
	if (!plan.feasible) return {};

	std::size_t relocated_count = 0;
	std::size_t bytes_relocated = 0;
	cudaStream_t stream = device_.managementStream();

	for (const CompactionPolicy::Move& move : plan.moves) {
		auto it = allocations_.find(move.block_id);
		if (it == allocations_.end() || it->second.in_use_count != 0) {
			// Can't actually happen (mutex_ has been held continuously since the
			// `movable` snapshot above) -- cheap insurance against ever acting
			// on a stale plan entry.
			continue;
		}

		awaitQuiescentLocked(move.block_id, lock);  // non-blocking here: in_use_count is already 0

		arena.claim(move.to);
		arena.relocate(move.from, move.to, stream);
		it->second.device_ptr = arena.pointerFor(move.to);
		it->second.unit_range = move.to;
		arena.free(move.from);

		++relocated_count;
		bytes_relocated += it->second.reserved_bytes;
	}

	return CompactionResult{relocated_count, bytes_relocated};
}

}  // namespace arachne::gpu
