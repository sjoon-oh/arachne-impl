#pragma once

#include <cstdint>

namespace arachne::gpu {

/// Access intent when resolving a StitchHandle to its device pointer --
/// currently informational only (both modes resolve identically today),
/// reserved for a future StitchPool that wants to distinguish read-shared
/// access from write-exclusive access at the memory layer itself,
/// mirroring the same distinction a Stitch's LeaseHandle already makes at
/// the policy level.
enum class AccessMode { Read, Write };

/// Which of DeviceContext's two pre-allocated pools an allocation is drawn
/// from (see gpu/device_context.hpp): `Data` is Anchor-driven,
/// promote/evict-eligible index/vector-data memory; `Metadata` is a
/// separate, physically distinct sub-region for state that needs
/// CPU/GPU synchronization but must never be evicted the way a Stitch's
/// index/vector data can be. Living in the same device pool as `Data`
/// (rather than being allocated ad hoc outside Arachne's own accounting)
/// is what keeps *all* GPU-resident Arachne memory under one budget;
/// keeping it in its own separate pool_memory_resource is what keeps its
/// fragmentation domain from mixing with Data's.
enum class MemoryKind { Data, Metadata };

/// Opaque identity for one variable-sized allocation a gpu::StitchPool
/// handed out. Never resolved to a pointer directly -- only
/// StitchPool::access() does that -- so that a Stitch
/// (core/anchor_manager.hpp) can carry one without anything that only
/// touches AnchorManager (e.g. its own unit tests) needing to link against
/// RAFT/RMM at all.
struct StitchHandle {
	std::uint64_t id = 0;
	bool valid() const { return id != 0; }
};

}  // namespace arachne::gpu
