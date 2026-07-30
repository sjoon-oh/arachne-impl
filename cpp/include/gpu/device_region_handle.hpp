#pragma once

#include <cstdint>

namespace arachne::gpu {

/// Which of DeviceContext's two pre-allocated pools an allocation is drawn
/// from (see gpu/device_context.hpp): `Data` is Anchor-driven,
/// promote/evict-eligible index/vector-data memory; `Metadata` is a
/// separate, physically distinct sub-region for state that needs
/// CPU/GPU synchronization but must never be evicted the way a Region's
/// index/vector data can be. Living in the same device pool as `Data`
/// (rather than being allocated ad hoc outside Arachne's own accounting)
/// is what keeps *all* GPU-resident Arachne memory under one budget;
/// keeping it in its own separate pool_memory_resource is what keeps its
/// fragmentation domain from mixing with Data's.
enum class MemoryKind { Data, Metadata };

/// Opaque identity for one variable-sized allocation a gpu::DeviceRegionPool
/// handed out. Never resolved to a pointer directly -- only
/// DeviceRegionPool::acquire() does that, via a Lease -- so that a Region
/// (core/region_manager.hpp) can carry one without anything that only
/// touches RegionManager (e.g. its own unit tests) needing to link against
/// RAFT/RMM at all.
struct DeviceRegionHandle {
	std::uint64_t id = 0;
	bool valid() const { return id != 0; }
};

}  // namespace arachne::gpu
