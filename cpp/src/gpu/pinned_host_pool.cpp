#include "gpu/pinned_host_pool.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace arachne::gpu {

PinnedHostPool::Buffer::~Buffer() { reset(); }

PinnedHostPool::Buffer::Buffer(Buffer&& other) noexcept
		: owner_(other.owner_), ptr_(other.ptr_), size_(other.size_) {
	other.owner_ = nullptr;
	other.ptr_ = nullptr;
	other.size_ = 0;
}

PinnedHostPool::Buffer& PinnedHostPool::Buffer::operator=(Buffer&& other) noexcept {
	if (this == &other) return *this;
	reset();
	owner_ = other.owner_;
	ptr_ = other.ptr_;
	size_ = other.size_;
	other.owner_ = nullptr;
	other.ptr_ = nullptr;
	other.size_ = 0;
	return *this;
}

void PinnedHostPool::Buffer::reset() {
	if (owner_ != nullptr && ptr_ != nullptr) owner_->release(ptr_, size_);
	owner_ = nullptr;
	ptr_ = nullptr;
	size_ = 0;
}

PinnedHostPool::~PinnedHostPool() {
	for (const auto& [bytes, ptr] : free_) cudaFreeHost(ptr);
}

PinnedHostPool::Buffer PinnedHostPool::acquire(std::size_t bytes) {
	if (bytes == 0) return {};
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = free_.lower_bound(bytes);
		if (it != free_.end()) {
			std::size_t capacity = it->first;
			void* ptr = it->second;
			free_.erase(it);
			cached_bytes_ -= capacity;
			return Buffer(*this, ptr, capacity);
		}
	}
	void* ptr = nullptr;
	cudaError_t status = cudaMallocHost(&ptr, bytes);
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("PinnedHostPool::cudaMallocHost: ") + cudaGetErrorString(status));
	}
	return Buffer(*this, ptr, bytes);
}

void PinnedHostPool::release(void* ptr, std::size_t bytes) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (bytes > cached_bytes_limit_ || cached_bytes_ > cached_bytes_limit_ - bytes) {
		cudaFreeHost(ptr);
		return;
	}
	free_.emplace(bytes, ptr);
	cached_bytes_ += bytes;
}

}  // namespace arachne::gpu
