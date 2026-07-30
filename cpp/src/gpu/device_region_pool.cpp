#include "gpu/device_region_pool.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "logging.hpp"

namespace arachne::gpu {

namespace {

void CheckCuda(cudaError_t status, const char* what) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
	}
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

DeviceRegionPool::DeviceRegionPool(DeviceContext& device) : device_(device) {}

DeviceRegionPool::~DeviceRegionPool() {
	// Safety net, not the expected path: every Region should have had its
	// device residency reclaimed via RegionManager::forget()/removeDependency()
	// (see core/region_manager.hpp) before the pool backing it goes away.
	// Reclaim whatever is still outstanding (memory and any leftover Lease
	// events) rather than leak.
	for (auto& [id, allocation] : allocations_) {
		for (auto& [stream, event] : allocation.last_used_events) {
			cudaEventDestroy(event);
		}
		resourceFor(allocation.kind)
				.deallocate(device_.resources().get_stream(), allocation.device_ptr, allocation.bytes,
										rmm::CUDA_ALLOCATION_ALIGNMENT);
	}
}

cuda::mr::any_resource<cuda::mr::device_accessible>& DeviceRegionPool::resourceFor(MemoryKind kind) {
	switch (kind) {
		case MemoryKind::Metadata:
			return device_.metadataResource();
		case MemoryKind::Data:
		default:
			return device_.dataResource();
	}
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

	// Every recorded event already happened in the past (its stream reached
	// that point) or will happen without anything further needed from us --
	// enqueueing a wait for each one on our own canonical stream makes
	// whatever we enqueue on that stream next (a deallocate or a
	// cudaMemcpyAsync) correctly ordered after all of them, without
	// host-blocking. Once enqueued, the dependency is permanent, so the
	// event objects themselves are no longer needed.
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
	void* ptr = resourceFor(kind).allocate(device_.resources().get_stream(), bytes,
																					rmm::CUDA_ALLOCATION_ALIGNMENT);

	std::lock_guard<std::mutex> lock(mutex_);
	std::uint64_t id = next_id_++;
	allocations_.emplace(id, Allocation{ptr, bytes, kind});
	return DeviceRegionHandle{id};
}

bool DeviceRegionPool::hasCapacity(std::size_t bytes, MemoryKind kind) const {
	return bytesAllocated(kind) + bytes <= device_.budgetBytes(kind);
}

std::optional<DeviceRegionHandle> DeviceRegionPool::tryAllocate(std::size_t bytes, MemoryKind kind) {
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

DeviceRegionPool::Lease DeviceRegionPool::acquire(DeviceRegionHandle handle, cudaStream_t stream) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allocations_.find(handle.id);
	if (it == allocations_.end()) {
		throw std::invalid_argument("DeviceRegionPool::acquire: handle is not a live allocation");
	}
	++it->second.in_use_count;
	return Lease(*this, handle, stream, it->second.device_ptr);
}

DeviceRegionPool::Lease DeviceRegionPool::acquire(DeviceRegionHandle handle) {
	return acquire(handle, device_.resources().get_stream().value());
}

void DeviceRegionPool::free(DeviceRegionHandle handle) {
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
	// Deallocate outside the lock: it's a real CUDA call, and nothing about
	// it needs allocations_ protection once this handle is already erased
	// from the map above.
	resourceFor(allocation.kind)
			.deallocate(device_.resources().get_stream(), allocation.device_ptr, allocation.bytes,
									rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void DeviceRegionPool::copyFromHost(DeviceRegionHandle handle, const void* host_src, std::size_t bytes,
																		 std::size_t dst_offset) {
	std::vector<Lease> pending;
	enqueueCopyFromHost(handle, host_src, bytes, dst_offset, pending);
	flush();
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

void DeviceRegionPool::flush() { device_.resources().sync_stream(); }

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

DeviceRegionPool::CompactionResult DeviceRegionPool::compact(MemoryKind kind) {
	if (device_.allocationPolicy() == AllocationPolicy::Naive) {
		// No shared arena under Naive -- see the declaration's doc comment.
		return {};
	}

	std::unique_lock<std::mutex> lock(mutex_);

	std::size_t live_bytes = 0;
	for (const auto& [id, allocation] : allocations_) {
		if (allocation.kind == kind) live_bytes += allocation.bytes;
	}
	if (live_bytes == 0) return {};

	struct Move {
		std::uint64_t id;
		void* new_ptr;
		void* old_ptr;
		std::size_t bytes;
	};
	std::vector<Move> moves;
	moves.reserve(allocations_.size());

	// Phase 1: for every live allocation of `kind`, wait out any outstanding
	// Lease on it (awaitQuiescentLocked -- the same mechanism free() uses),
	// then allocate a fresh block and enqueue a device-to-device copy on
	// DeviceContext's stream. All copies (and the waits enqueued ahead of
	// them) share that one stream, so they execute in enqueue order and a
	// single sync below is enough to know every one of them has landed.
	try {
		// Snapshot ids first: awaitQuiescentLocked can release/reacquire
		// `lock` while waiting, so iterating allocations_ directly while
		// calling it would risk iterator invalidation if anything else
		// mutates the map in that window.
		std::vector<std::uint64_t> candidates;
		for (const auto& [id, allocation] : allocations_) {
			if (allocation.kind == kind) candidates.push_back(id);
		}

		for (std::uint64_t id : candidates) {
			awaitQuiescentLocked(id, lock);

			auto it = allocations_.find(id);
			if (it == allocations_.end()) continue;  // freed while we waited -- nothing to move

			Allocation& allocation = it->second;
			void* new_ptr = resourceFor(kind).allocate(device_.resources().get_stream(), allocation.bytes,
																									 rmm::CUDA_ALLOCATION_ALIGNMENT);
			CheckCuda(cudaMemcpyAsync(new_ptr, allocation.device_ptr, allocation.bytes,
																 cudaMemcpyDeviceToDevice, device_.resources().get_stream().value()),
								"DeviceRegionPool::compact: cudaMemcpyAsync");
			moves.push_back(Move{id, new_ptr, allocation.device_ptr, allocation.bytes});
		}
	} catch (...) {
		// Unwind whatever we already allocated for this attempt before
		// propagating -- the original allocations are all still intact and
		// untouched (we haven't reached Phase 2 yet), so this is safe to bail
		// out of cleanly.
		for (const Move& move : moves) {
			resourceFor(kind).deallocate(device_.resources().get_stream(), move.new_ptr, move.bytes,
																		rmm::CUDA_ALLOCATION_ALIGNMENT);
		}
		throw;
	}

	device_.resources().sync_stream();

	// Phase 2: every copy has landed -- swap in the new pointers, then free
	// the old blocks.
	for (const Move& move : moves) {
		auto it = allocations_.find(move.id);
		if (it != allocations_.end()) it->second.device_ptr = move.new_ptr;
	}
	for (const Move& move : moves) {
		resourceFor(kind).deallocate(device_.resources().get_stream(), move.old_ptr, move.bytes,
																	rmm::CUDA_ALLOCATION_ALIGNMENT);
	}

	ARACHNE_LOG_DEBUG("DeviceRegionPool::compact: relocated {} allocation(s), {} bytes", moves.size(),
										 live_bytes);
	return CompactionResult{moves.size(), live_bytes};
}

}  // namespace arachne::gpu
