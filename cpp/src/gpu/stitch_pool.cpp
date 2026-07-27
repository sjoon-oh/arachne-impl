#include "gpu/stitch_pool.hpp"

#include <stdexcept>
#include <utility>

namespace arachne::gpu {

StitchPool::StitchPool(DeviceContext& device) : device_(device) {}

StitchPool::~StitchPool() {
	// Safety net, not the expected path: every Stitch should have been
	// free()'d via removeStitch()/forget() (see core/anchor_manager.hpp)
	// before the pool backing it goes away. Reclaim whatever is still
	// outstanding rather than leak device memory.
	for (auto& [id, allocation] : allocations_) {
		poolFor(allocation.kind)
				.deallocate(device_.resources().get_stream(), allocation.device_ptr, allocation.bytes,
										rmm::CUDA_ALLOCATION_ALIGNMENT);
	}
}

rmm::mr::pool_memory_resource& StitchPool::poolFor(MemoryKind kind) {
	switch (kind) {
		case MemoryKind::Metadata:
			return device_.metadataPool();
		case MemoryKind::Data:
		default:
			return device_.dataPool();
	}
}

StitchHandle StitchPool::allocate(std::size_t bytes, MemoryKind kind) {
	void* ptr = poolFor(kind).allocate(device_.resources().get_stream(), bytes,
																			rmm::CUDA_ALLOCATION_ALIGNMENT);

	std::lock_guard<std::mutex> lock(mutex_);
	std::uint64_t id = next_id_++;
	allocations_.emplace(id, Allocation{ptr, bytes, kind});
	return StitchHandle{id};
}

void* StitchPool::access(StitchHandle handle, AccessMode /*mode*/) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allocations_.find(handle.id);
	if (it == allocations_.end()) {
		throw std::invalid_argument("StitchPool::access: handle is not a live allocation");
	}
	return it->second.device_ptr;
}

void StitchPool::free(StitchHandle handle) {
	Allocation allocation;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = allocations_.find(handle.id);
		if (it == allocations_.end()) return;
		allocation = it->second;
		allocations_.erase(it);
	}
	// Deallocate outside the lock: it's a real CUDA call (potentially
	// blocking), and nothing about it needs allocations_ protection once this
	// handle is already erased from the map above.
	poolFor(allocation.kind)
			.deallocate(device_.resources().get_stream(), allocation.device_ptr, allocation.bytes,
									rmm::CUDA_ALLOCATION_ALIGNMENT);
}

std::size_t StitchPool::bytesAllocated() const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::size_t total = 0;
	for (const auto& [id, allocation] : allocations_) total += allocation.bytes;
	return total;
}

std::size_t StitchPool::bytesAllocated(MemoryKind kind) const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::size_t total = 0;
	for (const auto& [id, allocation] : allocations_) {
		if (allocation.kind == kind) total += allocation.bytes;
	}
	return total;
}

}  // namespace arachne::gpu
