#pragma once

// HnswIndexAnchorEntry: HnswIndexDist (hnsw_index_dist.hpp) plus two
// changes -- see the discussion this design came out of, summarized here:
//
//  1. Entry point: instead of always starting the level-0 walk at hnswlib's
//     raw global entry point (HnswIndex::resolveEntryPoint()'s default),
//     this class caches, per RoutingCache Anchor id, the internal id a
//     *completed* traversal for that Anchor actually landed on (its top
//     search result), and reuses that as the entry point the next time a
//     request carries the same TraverseRequest::anchor_id. Note the Anchor
//     id itself is *not* generally a real element of this index's own
//     dataset -- a Hybrid-triggered Anchor (Controller::search()'s fallback
//     path) is registered under the *query* vector, not a stored one (see
//     TraverseRequest::anchor_id's doc comment, adapter/index_adapter.hpp)
//     -- so anchor_entry_point_ below is populated from what a search
//     actually found, never by treating the Anchor id as one of this
//     index's own internal ids.
//  2. Beam width: BeamWidth() below is overridden to > 1 (see
//     HnswIndexDist::BeamWidth()'s doc comment), so TraverseOneOnDevice()
//     batches several round-1-eligible candidates into each GPU round-trip
//     instead of exactly one. This is the "(a)" option from the
//     dist-vs-anchor design discussion; "(b)" (gather a fixed N-hop
//     neighborhood up front and score it in one shot, abandoning strict
//     best-first order) was explicitly rejected as a *different* algorithm,
//     not implemented here.
//
// Both changes only affect traverseDevice() (inherited unmodified from
// HnswIndexDist, which is what actually calls resolveEntryPoint()/
// BeamWidth() -- both virtual, so overriding them here is enough, no need
// to re-override traverseDevice()/TraverseOneOnDevice() themselves).
// traverseHost() is overridden too, but only to populate
// anchor_entry_point_ as a side effect -- the search itself is still
// hnswlib's own searchKnnCloserFirst() (HnswIndex::traverseHost(),
// unmodified), which never consults resolveEntryPoint() at all (no seam to
// override its own internal entry point without patching hnswlib -- see
// hnsw_index.hpp's "used as-is" guarantee). So traverseHost()'s *results*
// for this class are always identical to HnswIndex's own, anchor_id or not
// -- only traverseDevice() ever actually benefits from the cache.
//
// anchor_entry_point_ is unbounded and never evicted (an Anchor id that's
// released on the Controller side has no corresponding notification here)
// -- acceptable for this experimental first cut, not solved here.

#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "hnsw_index_dist.hpp"

namespace arachne::index::hnsw {

class HnswIndexAnchorEntry final : public HnswIndexDist {
 public:
	/// Chosen to keep a single round-trip's combined neighbor batch
	/// (beam_width * hnswlib's level-0 M, typically a few hundred candidates)
	/// comfortably GPU-parallel-sized without exploding residency-scope risk
	/// (a wider beam visits more Regions per round, raising the chance one of
	/// them isn't GPU-resident -- see HnswIndexDist's file overview). Not
	/// tuned/benchmarked -- a reasonable starting point, not a measured
	/// optimum.
	static constexpr std::size_t kDefaultBeamWidth = 4;

	HnswIndexAnchorEntry(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
											 std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction,
											 std::size_t beam_width = kDefaultBeamWidth);
	~HnswIndexAnchorEntry() override = default;

	/// Identical results to HnswIndex::traverseHost() (see file overview --
	/// the host path never consults resolveEntryPoint()); overridden only to
	/// record, for each request carrying an anchor_id and returning at least
	/// one neighbor, that Anchor's entry-point cache entry -- see
	/// resolveEntryPoint() below for how it's consumed.
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override;

 protected:
	/// On a cache hit for request.anchor_id, returns the internal id a prior
	/// completed traversal for that Anchor actually landed on. Falls back to
	/// HnswIndex's own global-entry-point default otherwise (no anchor_id, or
	/// not cached yet -- e.g. this Anchor's first-ever GpuOnly attempt).
	std::uint32_t resolveEntryPoint(const TraverseRequest& request) const override;

	std::size_t BeamWidth() const override { return beam_width_; }

 private:
	std::size_t beam_width_;

	// Guards anchor_entry_point_ only -- deliberately separate from the
	// inherited mutex_ (HnswIndex's own doc comment): resolveEntryPoint() is
	// called from within TraverseOneOnDevice() while mutex_ is already held,
	// and traverseHost() releases mutex_ (via the base call) before touching
	// this map, so a single dedicated mutex avoids any reasoning about
	// re-entrant/nested locking against mutex_ altogether.
	mutable std::mutex anchor_cache_mutex_;
	std::unordered_map<VectorId, std::uint32_t> anchor_entry_point_;
};

}  // namespace arachne::index::hnsw
