#pragma once

#include <cstddef>
#include <map>
#include <mutex>

namespace arachne::gpu {

/// Reusable page-locked host staging. Buffers stay owned by a transfer batch
/// until its completion event fires, then return to this size-ordered cache.
class PinnedHostPool {
 public:
	class Buffer {
	 public:
		Buffer() = default;
		~Buffer();
		Buffer(Buffer&& other) noexcept;
		Buffer& operator=(Buffer&& other) noexcept;
		Buffer(const Buffer&) = delete;
		Buffer& operator=(const Buffer&) = delete;
		void* data() const { return ptr_; }

	 private:
		friend class PinnedHostPool;
		Buffer(PinnedHostPool& owner, void* ptr, std::size_t size) : owner_(&owner), ptr_(ptr), size_(size) {}
		void reset();
		PinnedHostPool* owner_ = nullptr;
		void* ptr_ = nullptr;
		std::size_t size_ = 0;
	};

	explicit PinnedHostPool(std::size_t cached_bytes_limit = std::size_t{1} << 26)
			: cached_bytes_limit_(cached_bytes_limit) {}
	~PinnedHostPool();
	Buffer acquire(std::size_t bytes);

 private:
	void release(void* ptr, std::size_t bytes);
	std::mutex mutex_;
	std::multimap<std::size_t, void*> free_;
	std::size_t cached_bytes_ = 0;
	std::size_t cached_bytes_limit_;
};

}  // namespace arachne::gpu
