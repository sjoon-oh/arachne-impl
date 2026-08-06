#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "gpu/device_context.hpp"
#include "gpu/device_region_handle.hpp"
#include "gpu/unit_pool_arena.hpp"

namespace arachne::gpu {

struct DeviceMemoryBlock {
	void* device_ptr = nullptr;
	std::size_t bytes = 0;
	std::size_t reserved_bytes = 0;
	MemoryKind kind = MemoryKind::Data;
	UnitPoolArena::UnitRange unit_range;
};

/// Storage-only interface. DeviceRegionPool retains handles, leases, transfer
/// ordering, and publication; allocation implementation is backend-private.
class DeviceMemoryManager {
 public:
	virtual ~DeviceMemoryManager() = default;
	virtual DeviceMemoryBlock allocate(std::size_t bytes, MemoryKind kind) = 0;
	virtual void deallocate(const DeviceMemoryBlock& block) = 0;
	virtual std::size_t reservationBytes(std::size_t bytes, MemoryKind kind) const = 0;
	virtual std::size_t budgetBytes(MemoryKind kind) const = 0;
	virtual std::size_t allocationUnitBytes(MemoryKind kind) const = 0;
	virtual UnitPoolArena* arena(MemoryKind) { return nullptr; }
};

/// One stream-ordered CUDA allocation per Region. No Arachne-owned arena or
/// fixed-size suballocation is shared with the Pooled implementation.
class AsyncDeviceMemoryManager final : public DeviceMemoryManager {
 public:
	explicit AsyncDeviceMemoryManager(DeviceContext& device) : device_(device) {}
	DeviceMemoryBlock allocate(std::size_t bytes, MemoryKind kind) override;
	void deallocate(const DeviceMemoryBlock& block) override;
	std::size_t reservationBytes(std::size_t bytes, MemoryKind kind) const override;
	std::size_t budgetBytes(MemoryKind kind) const override;
	std::size_t allocationUnitBytes(MemoryKind kind) const override;

 private:
	DeviceContext& device_;
};

/// Fixed UnitPoolArena implementation. It never calls the Async allocation
/// path and exposes its arena only for Pooled-only compaction.
class PooledDeviceMemoryManager final : public DeviceMemoryManager {
 public:
	explicit PooledDeviceMemoryManager(DeviceContext& device) : device_(device) {}
	DeviceMemoryBlock allocate(std::size_t bytes, MemoryKind kind) override;
	void deallocate(const DeviceMemoryBlock& block) override;
	std::size_t reservationBytes(std::size_t bytes, MemoryKind kind) const override;
	std::size_t budgetBytes(MemoryKind kind) const override;
	std::size_t allocationUnitBytes(MemoryKind kind) const override;
	UnitPoolArena* arena(MemoryKind kind) override;

 private:
	DeviceContext& device_;
};

std::unique_ptr<DeviceMemoryManager> MakeDeviceMemoryManager(DeviceContext& device);

}  // namespace arachne::gpu
