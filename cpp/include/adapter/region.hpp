#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace arachne {

/// Where a Region's authoritative mutable state currently lives.
enum class ResidencyState {
	Host,      // only the CPU-side index holds this region
	Pending,   // promotion/eviction transfer in flight
	Resident,  // materialized on GPU and eligible for GPU-only execution
};

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

struct ReconciliationReport {
	bool closed = false;  // true if boundary connectivity/invariants now hold
	RegionFootprint touched_neighbors;
};

/// Index-agnostic callback surface a Region implementation exposes to
/// Arachne's Core. Arachne never inspects index internals directly; it
/// only drives residency, leasing, and reconciliation through IRegion, so
/// any index can plug in as long as it can partition its state into objects
/// implementing this interface.
class IRegion {
 public:
	virtual ~IRegion() = default;

	virtual RegionId id() const = 0;
	virtual ResidencyState residency() const = 0;

	/// What this region covers. Used to compute Anchor footprints and
	/// drift/coverage statistics without Core understanding the
	/// underlying index structure.
	virtual RegionFootprint footprint() const = 0;

	/// Invoked by Core. Must bring the region's state onto GPU (or evict
	/// it) and update residency() accordingly.
	virtual void materializeOnDevice() = 0;
	virtual void evictFromDevice() = 0;

	/// Invoked by Core. acquireWriteLease grants this region GPU
	/// modification authority for one epoch; callers are expected to have
	/// already verified the region is Resident.
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
