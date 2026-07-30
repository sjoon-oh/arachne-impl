#pragma once

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapter/region.hpp"
#include "gpu/device_region_handle.hpp"
#include "types.hpp"

namespace arachne {

/// The single physical-mapping record for one RegionId: where its data
/// lives in host memory (`host`, reported by the adapter via
/// IRegion::hostView() -- see adapter/region.hpp), where it lives on GPU
/// once promoted (`device`, valid only after Controller actually allocates
/// through gpu::DeviceRegionPool), and the GPU write authority currently
/// held over it (`lease`, valid only while promoted). There is exactly one
/// Region per RegionId inside RegionManager -- unlike the old per-Anchor
/// Stitch, multiple Anchors that depend on the same Region all see the
/// same host/device/lease state, because there is only one copy of it.
struct Region {
	RegionId id = 0;
	HostRegionView host;
	gpu::DeviceRegionHandle device;
	LeaseHandle lease;
};

/// Owns every registered Region plus the bipartite Anchor<->Region
/// dependency graph, replacing the old AnchorManager/Stitch pair. Two
/// things used to live awkwardly inside `Stitch`: "where is this Region's
/// data" (a physical fact, true regardless of which Anchor is asking) and
/// "which Anchor currently depends on it" (a membership fact). Splitting
/// them means promoting a Region once and letting many Anchors share that
/// promotion, rather than each Anchor separately negotiating its own
/// lease/copy of what is physically the same GPU memory.
///
/// registerRegion() is the adapter's explicit opt-in: a Region only
/// becomes promotion/eviction-eligible once its owning index calls it
/// (directly or, today, via Controller -- see Controller::registerRegion())
/// to declare "this host-resident data is a candidate for GPU residency"
/// and report where it lives. Data the adapter never registers stays
/// entirely outside Arachne's accounting, exactly like before registration
/// existed.
///
/// Thread-safe: every method takes RegionManager's own lock, mirroring
/// AnchorManager's old concurrency contract (Controller is called
/// concurrently).
class RegionManager {
 public:
	/// Registers `id` as promotion/eviction-eligible and records `host` as
	/// where its data currently lives. No-op if `id` is already registered --
	/// re-registering does not refresh `host`; a moved host allocation is out
	/// of scope for this skeleton (see IRegion::hostView()'s doc comment).
	void registerRegion(RegionId id, HostRegionView host);

	bool isRegistered(RegionId id) const;

	/// Snapshot of `id`'s current Region record. Throws std::invalid_argument
	/// if `id` was never registered.
	Region regionOf(RegionId id) const;

	/// RegionIds `anchor_id` currently depends on (empty if none).
	std::vector<RegionId> regionsOf(VectorId anchor_id) const;

	/// Records `anchor_id` as a dependent of `region_id` (idempotent -- a
	/// second call for the same pair changes nothing). Returns false without
	/// recording anything if `region_id` was never registered -- callers
	/// (Controller::make()) are expected to check this before attempting to
	/// promote a region nobody opted in.
	bool addDependency(VectorId anchor_id, RegionId region_id);

	/// Drops `anchor_id`'s dependency on `region_id`. Returns true exactly
	/// when that was the *last* Anchor depending on it -- the caller's signal
	/// to actually reclaim the lease/device residency via clearResidency()
	/// below, since other Anchors may still be relying on this Region staying
	/// promoted. Returns false (and changes nothing) if there was no such
	/// dependency, or if other Anchors still depend on `region_id`.
	bool removeDependency(VectorId anchor_id, RegionId region_id);

	/// Drops every dependency `anchor_id` has, returning the RegionIds that
	/// consequently dropped to zero dependents -- the set the caller should
	/// reclaim (release lease, free device memory, clearResidency()). Regions
	/// still depended on by some other Anchor are not included, matching
	/// removeDependency()'s per-pair semantics.
	std::vector<RegionId> forget(VectorId anchor_id);

	/// Updates region `id`'s lease field, e.g. right after Controller calls
	/// IRegion::acquireWriteLease() for it. No-op if `id` isn't registered.
	void setLease(RegionId id, LeaseHandle lease);

	/// Updates region `id`'s device field, e.g. right after Controller
	/// allocates GPU memory for it through gpu::DeviceRegionPool. No-op if
	/// `id` isn't registered.
	void setDevice(RegionId id, gpu::DeviceRegionHandle device);

	/// Resets region `id` back to host-only (invalid lease and device),
	/// leaving it registered (host mapping untouched) -- called once the
	/// caller has actually released the write lease / freed the device
	/// memory a removeDependency()/forget() result told it to reclaim. No-op
	/// if `id` isn't registered.
	void clearResidency(RegionId id);

 private:
	mutable std::mutex mutex_;
	std::unordered_map<RegionId, Region> regions_;
	std::unordered_map<RegionId, std::unordered_set<VectorId>> dependents_;
	std::unordered_map<VectorId, std::unordered_set<RegionId>> dependencies_;
};

}  // namespace arachne
