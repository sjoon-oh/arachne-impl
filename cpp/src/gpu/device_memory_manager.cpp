#include "gpu/device_memory_manager.hpp"

#include <stdexcept>
#include <string>

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

DeviceMemoryBlock AsyncDeviceMemoryManager::allocate(std::size_t bytes, MemoryKind kind) {
	void* ptr = nullptr;
	CheckCuda(cudaMallocAsync(&ptr, bytes, device_.managementStream()),
					"AsyncDeviceMemoryManager::cudaMallocAsync");
	return DeviceMemoryBlock{ptr, bytes, bytes, kind, {}};
}

void AsyncDeviceMemoryManager::deallocate(const DeviceMemoryBlock& block) {
	if (block.device_ptr == nullptr) return;
	CheckCuda(cudaFreeAsync(block.device_ptr, device_.managementStream()),
					"AsyncDeviceMemoryManager::cudaFreeAsync");
}

std::size_t AsyncDeviceMemoryManager::reservationBytes(std::size_t bytes, MemoryKind) const { return bytes; }
std::size_t AsyncDeviceMemoryManager::budgetBytes(MemoryKind kind) const { return device_.budgetBytes(kind); }
std::size_t AsyncDeviceMemoryManager::allocationUnitBytes(MemoryKind) const { return 1; }

UnitPoolArena* PooledDeviceMemoryManager::arena(MemoryKind kind) {
	return kind == MemoryKind::Metadata ? device_.metadataArena() : device_.dataArena();
}

DeviceMemoryBlock PooledDeviceMemoryManager::allocate(std::size_t bytes, MemoryKind kind) {
	UnitPoolArena* target = arena(kind);
	if (target == nullptr) throw std::logic_error("PooledDeviceMemoryManager: missing arena");
	std::uint64_t units = RequiredUnits(bytes, target->unitBytes());
	auto range = target->allocateBestFit(units);
	if (!range.has_value()) {
		throw std::runtime_error("PooledDeviceMemoryManager: insufficient contiguous arena space");
	}
	return DeviceMemoryBlock{target->pointerFor(*range), bytes, units * target->unitBytes(), kind, *range};
}

void PooledDeviceMemoryManager::deallocate(const DeviceMemoryBlock& block) {
	UnitPoolArena* target = arena(block.kind);
	if (target == nullptr) throw std::logic_error("PooledDeviceMemoryManager: missing arena");
	target->free(block.unit_range);
}

std::size_t PooledDeviceMemoryManager::reservationBytes(std::size_t bytes, MemoryKind kind) const {
	UnitPoolArena* target = kind == MemoryKind::Metadata ? device_.metadataArena() : device_.dataArena();
	return RequiredUnits(bytes, target->unitBytes()) * target->unitBytes();
}

std::size_t PooledDeviceMemoryManager::budgetBytes(MemoryKind kind) const { return device_.budgetBytes(kind); }

std::size_t PooledDeviceMemoryManager::allocationUnitBytes(MemoryKind kind) const {
	UnitPoolArena* target = kind == MemoryKind::Metadata ? device_.metadataArena() : device_.dataArena();
	return target->unitBytes();
}

std::unique_ptr<DeviceMemoryManager> MakeDeviceMemoryManager(DeviceContext& device) {
	if (device.allocationPolicy() == AllocationPolicy::Pooled) {
		return std::make_unique<PooledDeviceMemoryManager>(device);
	}
	return std::make_unique<AsyncDeviceMemoryManager>(device);
}

}  // namespace arachne::gpu
