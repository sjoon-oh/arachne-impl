#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "adapter/region.hpp"
#include "gpu/stitch_handle.hpp"
#include "types.hpp"

namespace arachne {

/// A Stitch is the association between an Anchor (identified only by its
/// VectorId -- RoutingCache doesn't know or care what a Stitch is) and a
/// Region that Anchor currently holds GPU write authority (a Lease) over,
/// plus the GPU-resident memory backing that authority. `memory` is
/// obtained from a gpu::StitchPool (see gpu/stitch_pool.hpp) -- Arachne,
/// not the adapter, owns this allocation, per the Anchor-centric residency
/// policy (variable-sized, keyed by handle rather than by fixed address).
/// Left invalid ({} / StitchHandle{}) for callers that don't yet wire a
/// StitchPool through -- AnchorManager itself never allocates or frees it,
/// only carries it.
struct Stitch {
	RegionId region = 0;
	LeaseHandle lease;
	gpu::StitchHandle memory;
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
	/// `region`. `memory` defaults to an invalid handle for callers that
	/// don't yet allocate GPU memory for a Stitch through a gpu::StitchPool.
	void addStitch(VectorId anchor_id, RegionId region, LeaseHandle lease,
								gpu::StitchHandle memory = {});

	/// Removes the Stitch for `region`, if any, returning it so the caller
	/// can release both the underlying IRegion lease and (via
	/// gpu::StitchPool::free()) the GPU allocation backing `memory`. Returns
	/// a default-constructed Stitch (invalid lease) and does nothing if there
	/// was nothing to remove.
	Stitch removeStitch(VectorId anchor_id, RegionId region);

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
