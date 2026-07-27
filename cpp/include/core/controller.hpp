#pragma once

#include <optional>
#include <cstdint>

#include "adapter/index_adapter.hpp"
#include "core/anchor_manager.hpp"
#include "core/op_scheduler.hpp"
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
	Controller(IndexAdapter& adapter, RoutingCache& routing_cache,
			 SchedulingConfig scheduling_config = {});

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
	// Distinct from the Stitch-based Lease replacement policy below: this is
	// general read-side residency, not write-lease bookkeeping.
	void promote(const RegionFootprint& hot_host_footprint);
	void evict(const RegionFootprint& candidates);

	// Selective verification (design point 3). Not yet wired into search().
	// On mismatch, reclaims every Stitch on `anchor_id` (via anchor_manager_)
	// since the regions it currently points at no longer represent its
	// locality.
	void verify(const Query& query, VectorId anchor_id, const TraverseResult& gpu_only_result);

	// Stitch lifecycle (design point 4): acquires a write lease on `region`
	// and records it against `anchor_id` via anchor_manager_. The eventual
	// Anchor-recency replacement policy (reclaim every Stitch on a cold
	// Anchor) would go through anchor_manager_.forget() directly rather than
	// a Core-level counterpart to this.
	//
	// Does not yet allocate GPU memory for the Stitch via stitch_pool_ above
	// -- that needs a way to learn how many bytes `region` needs, which
	// IRegion doesn't expose yet. Once it does, this is where
	// stitch_pool_.allocate() gets called and the resulting handle threaded
	// into anchor_manager_.addStitch()'s `memory` parameter; the mirror image
	// (stitch_pool_.free()) belongs wherever a Stitch is removed --
	// currently just the `stale` loop in verify() below.
	bool make(VectorId anchor_id, RegionId region);

	IndexAdapter& adapter_;
	RoutingCache& routing_cache_;
	OpScheduler scheduler_;
	AnchorManager anchor_manager_;
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
