#pragma once

#include <cstdint>

#include "arachne/adapter/index_adapter.hpp"
#include "arachne/core/anchor_manager.hpp"
#include "arachne/core/routing_cache.hpp"

namespace arachne {

/// Arachne's core management engine: the index-agnostic control plane that
/// decides where SEARCH/INSERT/DELETE run and how GPU residency/write
/// authority is managed, per the Quick Summary. This is the class meant to
/// carry Arachne's actual design as it gets built out; everything here is
/// implemented against IndexAdapter/IRegion and RoutingCache only, never
/// against a concrete index or a concrete Anchor storage structure.
///
/// Splits "is this query close to something we've seen" (RoutingCache,
/// injected -- pure identity/routing signal, pluggable) from "what write
/// leases has that Anchor earned" (AnchorManager, owned directly -- Core's
/// own policy-state, not pluggable). Core is the only thing that talks to
/// both.
class Core {
 public:
	Core(IndexAdapter& adapter, RoutingCache& routing_cache);

	SearchResult search(const Query& query);
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

 private:
	// Anchor Query routing (design point 1).
	struct RoutingDecision {
		bool gpu_only_eligible = false;
		RegionFootprint predicted_scope;  // derived from the matched Anchor's Stitches
	};
	RoutingDecision route(const Query& query) const;

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
	void verify(const Query& query, VectorId anchor_id, const TraversalResult& gpu_only_result);

	// Stitch lifecycle (design point 4): acquires a write lease on `region`
	// and records it against `anchor_id` via anchor_manager_. The eventual
	// Anchor-recency replacement policy (reclaim every Stitch on a cold
	// Anchor) would go through anchor_manager_.forget() directly rather than
	// a Core-level counterpart to this.
	bool make(VectorId anchor_id, RegionId region);

	IndexAdapter& adapter_;
	RoutingCache& routing_cache_;
	AnchorManager anchor_manager_;
	VectorId next_anchor_id_ = 1;
	std::uint64_t drift_window_host_ = 0;
	std::uint64_t drift_window_total_ = 0;
};

}  // namespace arachne
