#pragma once

#include <optional>
#include <cstdint>
#include <memory>

#include "adapter/index_adapter.hpp"
#include "core/anchor_manager.hpp"
#include "core/op_scheduler.hpp"
#include "core/replacement_policy.hpp"
#include "core/routing_cache.hpp"

#if defined(ARACHNE_WITH_RAFT)
#include "gpu/device_context.hpp"
#include "gpu/stitch_pool.hpp"
#endif

namespace arachne {

/// Arachne's core management controller: the index-agnostic control plane that
/// decides where SEARCH/INSERT/DELETE run and how GPU residency/write
/// authority is managed, per the Quick Summary. This is the class meant to
/// carry Arachne's actual design as it gets built out; everything here is
/// implemented against IndexAdapter/IRegion and RoutingCache only, never
/// against a concrete index or a concrete Anchor storage structure.
///
/// Splits "is this query close to something we've seen" (RoutingCache,
/// injected -- pure identity/routing signal, pluggable) from "what write
/// leases has that Anchor earned" (AnchorManager, owned directly -- Controller's
/// own policy-state, not pluggable). Controller is the only thing that talks to
/// both.
class Controller {
 public:
	// `replacement_policy` defaults to FifoReplacementPolicy when left null,
	// mirroring OpScheduler's own SchedulingPolicy default (see
	// core/op_scheduler.hpp) -- see ReplacementPolicy's doc comment.
	Controller(IndexAdapter& adapter, RoutingCache& routing_cache,
			 SchedulingConfig scheduling_config = {},
			 std::unique_ptr<ReplacementPolicy> replacement_policy = nullptr);

	SearchResult search(const Query& query);
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

 private:
	struct SearchPlan {
		TraverseRequest primary;
		bool fallback_to_hybrid = false;
	};

	struct InsertPlan {
		VectorId anchor_id = 0;
		ModifyRequest request;
	};

	struct RemovePlan {
		ModifyRequest request;
	};

	// Anchor Query routing (design point 1).
	struct RoutingDecision {
		bool gpu_only = false;
		RegionFootprint predicted_scope;  // derived from the matched Anchor's Stitches
	};
		RoutingDecision route(const Query& query);
		SearchPlan routeSearch(const Query& query);
		InsertPlan routeInsert(const Record& record);
		RemovePlan routeRemove(VectorId id);

	TraverseResult dispatch(const TraverseRequest& request);
	ModifyResult dispatch(const ModifyRequest& request);

	SearchResult commitSearch(const SearchPlan& plan, const TraverseResult& result,
													 bool final_was_hybrid);
	InsertResult commitInsert(const InsertPlan& plan, const ModifyResult& result);
	DeleteResult commitRemove(const RemovePlan& plan, const ModifyResult& result);

	// Workload drift (design point 2 trigger).
	void recordTraversalForDrift(bool touched_host);

	// Promotion / eviction (design point 2). Not yet wired to a drift
	// trigger -- the policy for "when" is still open; these are the "how".
	// Distinct from the Anchor/Stitch-based replacement policy below: this is
	// general read-side residency, not write-lease bookkeeping.
	void promote(const RegionFootprint& hot_host_footprint);
	void evict(const RegionFootprint& candidates);

	// Anchor-centric Promotion (design point 4): grants `anchor_id` write-lease
	// Stitches over every region in `footprint`. 1) registers `anchor_id` in
	// routing_cache_ under `anchor_vector` so future queries route to it, 2)
	// calls make() per region, which materializes the region on device (see
	// IRegion::materializeOnDevice()) and acquires a write lease for it, and
	// 3) if make() fails for a region, asks replacement_policy_ for an Anchor
	// to reclaim (excluding `anchor_id` itself), evicts it via evictAnchor(),
	// and retries that one region once. Gives up on a region (leaving it
	// un-stitched for this call) if there's no eviction candidate or the
	// retry still fails -- a future call can try again.
	void promoteAnchor(VectorId anchor_id, const VectorView& anchor_vector,
									const RegionFootprint& footprint);

	// Reclaims every Stitch currently held by `anchor_id`: releases each
	// write lease, frees the GPU memory backing it (once StitchPool sizing is
	// wired -- see make()'s doc comment), and notifies replacement_policy_ so
	// it stops tracking `anchor_id`. The mechanism promoteAnchor() calls
	// through replacement_policy_->selectEvictionCandidate() when it needs
	// room, and the one verify() below uses on a mismatch.
	void evictAnchor(VectorId anchor_id);

	// Selective verification (design point 3). Not yet wired into search().
	// On mismatch, reclaims every Stitch on `anchor_id` (via anchor_manager_)
	// since the regions it currently points at no longer represent its
	// locality.
	void verify(const Query& query, VectorId anchor_id, const TraverseResult& gpu_only_result);

	// Stitch lifecycle (design point 4): materializes `region` on device if
	// it isn't already (IRegion::materializeOnDevice()), acquires a write
	// lease on it, and records both against `anchor_id` via anchor_manager_
	// and replacement_policy_ (see promoteAnchor(), the caller this exists
	// for). Returns false, leaving no new state behind, if `region` doesn't
	// resolve, doesn't become resident, or isn't lease-eligible right now --
	// callers wanting eviction-and-retry on failure go through
	// promoteAnchor(), not this directly.
	//
	// Does not yet allocate GPU memory for the Stitch via stitch_pool_ above
	// -- that needs a way to learn how many bytes `region` needs, which
	// IRegion doesn't expose yet. Once it does, this is where
	// stitch_pool_.allocate() gets called and the resulting handle threaded
	// into anchor_manager_.addStitch()'s `memory` parameter; the mirror image
	// (stitch_pool_.free()) already happens in evictAnchor() below, guarded
	// on Stitch::memory.valid() so it's a no-op until that wiring exists.
	bool make(VectorId anchor_id, RegionId region);

	IndexAdapter& adapter_;
	RoutingCache& routing_cache_;
	OpScheduler scheduler_;
	AnchorManager anchor_manager_;
	// Strategy (design point 4): see ReplacementPolicy's doc comment.
	std::unique_ptr<ReplacementPolicy> replacement_policy_;
#if defined(ARACHNE_WITH_RAFT)
	// GPU residency accounting (design point 4): Arachne-owned, not the
	// adapter's -- see gpu/device_context.hpp and gpu/stitch_pool.hpp. Not
	// yet wired into make()/verify() below (see their doc comments); this is
	// the seam for that once a Stitch's byte size can be determined (needs
	// an IRegion sizing hook that doesn't exist yet).
	gpu::DeviceContext device_;
	gpu::StitchPool stitch_pool_;
#endif
	VectorId next_anchor_id_ = 1;
	std::uint64_t drift_window_host_ = 0;
	std::uint64_t drift_window_total_ = 0;
};

}  // namespace arachne
