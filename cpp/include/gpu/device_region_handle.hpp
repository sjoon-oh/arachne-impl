#pragma once

#include <cstdint>

namespace arachne::gpu {

/// Identity types shared between DeviceContext, DeviceRegionPool, and
/// RegionManager for GPU-resident memory.
///
/// MemoryKind selects one of DeviceContext's two pre-allocated pools (see
/// gpu/device_context.hpp): `Data` is Anchor-driven, promote/evict-eligible
/// index/vector-data memory; `Metadata` is a separate, physically distinct
/// sub-region for state that needs CPU/GPU synchronization but must never
/// be evicted the way a Region's index/vector data can be. Both live under
/// one Arachne-wide memory budget (rather than either being allocated ad
/// hoc outside Arachne's own accounting), but stay in separate
/// arenas/resources so their fragmentation domains never mix.
///
/// DeviceRegionHandle is the opaque identity gpu::DeviceRegionPool hands
/// out for one variable-sized allocation. It is never resolved to a
/// pointer directly -- only DeviceRegionPool::acquire() does that, via a
/// Lease -- so that a Region (core/region_manager.hpp) can carry a handle
/// without anything that only touches RegionManager (e.g. its own unit
/// tests) needing to link against RAFT/RMM at all.
enum class MemoryKind { Data, Metadata };

struct DeviceRegionHandle {
	std::uint64_t id = 0;
	bool valid() const { return id != 0; }
};

}  // namespace arachne::gpu
