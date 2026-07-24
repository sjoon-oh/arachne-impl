#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "arachne/adapter/region.hpp"
#include "arachne/types.hpp"

namespace arachne {

/// A Stitch is the association between an Anchor (identified only by its
/// VectorId -- RoutingCache doesn't know or care what a Stitch is) and a
/// Region that Anchor currently holds GPU write authority (a Lease) over.
struct Stitch {
	RegionId region = 0;
	LeaseHandle lease;
};

/// Owns Stitch (GPU write-lease) bookkeeping for whatever Anchor ids Core
/// hands it. Deliberately separate from RoutingCache: RoutingCache answers
/// only "is this query close to something we've seen" (the Quick Summary's
/// pre-filter question, design point 1), while write-lease state --
/// design point 4, and eventually the replacement policy for which
/// Anchors' Stitches get reclaimed -- lives here instead. Thread-safe,
/// since Core is called concurrently the same way RoutingCache is.
class AnchorManager {
 public:
	/// Copy of the Stitches currently held by `anchor_id` (empty if none).
	std::vector<Stitch> stitchesOf(VectorId anchor_id) const;

	/// Records a new Stitch. No-op if `anchor_id` already has one for
	/// `region`.
	void addStitch(VectorId anchor_id, RegionId region, LeaseHandle lease);

	/// Removes the Stitch for `region`, if any, returning its LeaseHandle so
	/// the caller can release the underlying IRegion lease. Returns an
	/// invalid handle (and does nothing) if there was nothing to remove.
	LeaseHandle removeStitch(VectorId anchor_id, RegionId region);

	/// Removes every Stitch recorded for `anchor_id` and returns them, so
	/// the caller can release each underlying lease. The hook a future
	/// eviction policy reclaiming a cold Anchor entirely would use, rather
	/// than reclaiming one Region's Stitch at a time via removeStitch().
	std::vector<Stitch> forget(VectorId anchor_id);

 private:
	mutable std::mutex mutex_;
	std::unordered_map<VectorId, std::vector<Stitch>> stitches_;
};

}  // namespace arachne
