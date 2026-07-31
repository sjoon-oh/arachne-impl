#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace arachne {

/// The set of regions an operation touched, or is scoped to.
struct RegionFootprint {
	std::vector<RegionId> regions;
};

/// A GPU write lease grants a Region temporary modification authority, per
/// Quick Summary design point 4 (GPU Write Lease). Opaque outside Core
/// and the Region implementation that issued it.
struct LeaseHandle {
	RegionId region = 0;
	std::uint64_t epoch = 0;
	bool valid() const { return epoch != 0; }
};

/// Index-specific encoding of what changed inside a Region during a lease
/// epoch (e.g. inserted rows, edge rewrites). Left opaque at the Core
/// level; adapters interpret their own deltas.
struct ModificationDelta {
	std::vector<std::byte> payload;
};

/// Where a Region's data currently lives in host memory: a pointer/size
/// pair the adapter reports and Arachne only ever records, never allocates
/// or frees -- host memory stays entirely the index's own to manage (see
/// IRegion::hostView()). Paired with gpu::DeviceRegionHandle (see
/// gpu/device_region_handle.hpp) inside a Region (core/region_manager.hpp):
/// together the two describe where one Region's data lives on each side of
/// the host/device boundary.
///
/// `subregion_bytes` sets the granularity at which Core tracks which parts
/// of this Region were written to during a GPU write lease (see
/// gpu/dirty_header.hpp for the bitmap this drives). The adapter, not Core,
/// picks the value, since it should match the index's own natural write
/// unit (e.g. roughly one graph node's record for an HNSW-style adapter). 0
/// (the default) disables fine-grained tracking and treats the whole Region
/// as one dirty/clean unit -- the only real choice until Core's GPU
/// allocation for Regions is wired (see Controller::make()'s doc comment).
struct HostRegionView {
	void* ptr = nullptr;
	std::size_t bytes = 0;
	std::size_t subregion_bytes = 0;
};

struct ReconciliationReport {
	bool closed = false;  // true if boundary connectivity/invariants now hold
	RegionFootprint touched_neighbors;
};

/// Index-agnostic callback surface a Region implementation exposes to
/// Arachne's Core. Arachne never inspects index internals directly; it
/// only drives leasing and reconciliation through IRegion, so any index can
/// plug in as long as it can partition its state into objects implementing
/// this interface. GPU residency itself -- whether/when a Region's bytes are
/// actually copied onto or off of the device -- is entirely Arachne's own
/// decision and mechanism (see core/controller.hpp's Controller::make()/
/// evictAnchor()), not something an adapter opts into or performs; the
/// adapter's only role here is to declare a Region exists and hand out
/// leases over it.
class IRegion {
 public:
	virtual ~IRegion() = default;

	virtual RegionId id() const = 0;

	/// What this region covers. Used to compute Anchor footprints and
	/// drift/coverage statistics without Core understanding the
	/// underlying index structure.
	virtual RegionFootprint footprint() const = 0;

	/// Where this region's authoritative data currently lives in host
	/// memory. Arachne records the pointer/size purely as a mapping (see
	/// core/region_manager.hpp) -- never dereferences, allocates, or frees
	/// it; the adapter may change its own host layout as long as this stays accurate.
	virtual HostRegionView hostView() const = 0;

	/// Invoked by Core. acquireWriteLease grants this region GPU
	/// modification authority for one epoch; callers are expected to have
	/// already verified the region is GPU-resident (Core's own
	/// RegionManager-tracked state, not anything this interface reports).
	virtual LeaseHandle acquireWriteLease() = 0;
	virtual void releaseWriteLease(LeaseHandle handle) = 0;

	/// Applies a modification produced by a traverse+modify pair while the
	/// caller holds `handle`. A full implementation accumulates dirty state
	/// for the lease epoch rather than writing back to Host immediately.
	virtual void applyLocalModification(LeaseHandle handle, const ModificationDelta& delta) = 0;

	/// Background reconciliation from Quick Summary design point 4: repairs
	/// connectivity/invariants with regions this one does not own. Called
	/// after eviction, dirty-state pressure, lease revoke, or a maintenance
	/// trigger.
	virtual ReconciliationReport reconcileBoundary() = 0;
};

}  // namespace arachne
