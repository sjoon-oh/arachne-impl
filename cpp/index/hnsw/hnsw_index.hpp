#pragma once

// HnswIndex: an IAdapter over thirdparty/hnswlib's HierarchicalNSW graph --
// the real, production ANN index report.md's Phase 1 (§7.2) plans to wire
// up, as opposed to test/stress/stress_index.hpp's brute-force test double.
//
// HnswIndex itself is CONCRETE (not abstract) -- it is, on its own, the
// plain/original hnswlib variant: always hnswlib's own global entry point +
// full upper-level descent, exactly like unmodified hnswlib (report.md §10,
// "옵션 2"). Two sibling classes extend it with alternate GPU strategies,
// kept as separate classes (not runtime toggles) so each can be
// built/benchmarked/tested independently -- see the discussion that led
// here (the same "measure, don't assume" lesson report.md §5 drew from
// SVFusion's own offload experiment):
//   - HnswIndexDist (hnsw_index_dist.hpp): offloads per-hop candidate
//     distance computation to a GPU kernel during traverseDevice(), control
//     flow (candidate queue, visited set, stopping condition) stays on
//     host. Report.md §10, "옵션 1" for the *device* path -- not to be
//     confused with resolveEntryPoint() below, which is about host entry
//     point selection, a separate question.
//   - HnswIndexAnchorEntry (hnsw_index_anchor_entry.hpp): opportunistically
//     skips the upper-level descent via a cached Anchor id -> internal id,
//     overriding resolveEntryPoint() below (report.md §10.4).
//
// Host buffer / Region layout (report.md §7.2, §2.1):
// HnswEngine (hnsw_index.cpp, type-erased over the concrete
// hnswlib::HierarchicalNSW<DistT>) owns one contiguous data_level0_memory_
// block -- exactly the layout StressIndex's buffer_ mirrors, one fixed-size
// record per internal id (level-0 links + vector + label, see
// thirdparty/hnswlib/hnswlib/hnswalg.h). HnswIndex slices that block into
// equal id-range spans of `vectors_per_region` records each, one HnswRegion
// per span, matching StressIndex's scheme so Arachne's Region machinery is
// exercised identically. capacity_ is fixed at construction --
// hnswlib::HierarchicalNSW::resizeIndex() is never called (report.md §7.2:
// its realloc() would invalidate any host pointer already promoted to
// GPU).
//
// hnswlib itself is used entirely as-is: HnswEngine's TypedHnswEngine
// (hnsw_index.cpp) calls only hnswlib's own public API/public members
// (searchKnnCloserFirst(), addPoint(), markDelete(), saveIndex(),
// loadIndex(), label_lookup_, ...) -- thirdparty/hnswlib is never patched or
// forked for this class. HnswIndexDist is the one exception in this
// directory (see its own file for what and why it copies).
//
// Known open problem carried over from report.md §10.3, not solved here:
// hnswlib internal ids are assigned in *insertion* order, not spatial
// order, so an id-contiguous Region does not necessarily correspond to a
// graph-local neighborhood. Left as-is per report.md's decision to treat
// this as a separate, later concern (§10.5).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "adapter/region.hpp"
#include "core/routing_cache.hpp"  // DistanceMetric
#include "types.hpp"

namespace arachne {
class Controller;
}

namespace arachne::index::hnsw {

/// One id-range slice of the underlying hnswlib graph's level-0 record
/// array -- HnswIndex's Region unit. Structurally identical to
/// test/stress/stress_index.hpp's StressRegion; the only difference is
/// which adapter's memory it points into.
class HnswRegion final : public IRegion {
 public:
	HnswRegion(RegionId id, void* ptr, std::size_t bytes, std::size_t subregion_bytes);

	RegionId id() const override { return id_; }
	RegionFootprint footprint() const override { return RegionFootprint{{id_}}; }
	HostRegionView hostView() const override { return host_; }
	LeaseHandle acquireWriteLease() override { return LeaseHandle{id_, ++epoch_}; }
	void releaseWriteLease(LeaseHandle) override {}

	/// TODO(Phase 1, report.md §7.1/§10 decision 3): once GPU write leases are
	/// actually exercised against a real hnswlib-backed Region, this needs to
	/// interpret `delta` the way mutuallyConnectNewElement()'s rewiring would
	/// (see report.md §3 decision 3 / §10 for the cross-region write
	/// discussion). No-op for now, matching StressRegion's own placeholder.
	void applyLocalModification(LeaseHandle, const ModificationDelta&) override {}

	/// TODO(report.md §3 decision 3): background boundary reconciliation
	/// across Regions -- not yet implemented.
	ReconciliationReport reconcileBoundary() override { return ReconciliationReport{}; }

 private:
	RegionId id_;
	HostRegionView host_;
	std::uint64_t epoch_ = 0;
};

/// Type-erased handle onto one concrete hnswlib::HierarchicalNSW<DistT>
/// instantiation. Solves the same (SpaceT, ElemT, DistT) dispatch problem
/// ASRoutingCacheHnsw::TypedInstance (src/core/as_routing_cache_hnsw.cpp)
/// already solves for the routing-cache's small Anchor index -- reused here
/// because HnswIndex needs the identical (VectorDType, DistanceMetric) ->
/// concrete hnswlib type matrix, just for the full dataset instead of a
/// small Anchor set. Defined in hnsw_index.cpp (not this header) so hnswlib
/// stays a PRIVATE, impl-only dependency of this adapter -- same reasoning
/// as core/routing_cache_hnsw.hpp's own forward-declared Instance type.
class HnswEngine;

/// IAdapter over hnswlib's HierarchicalNSW graph, as-is (see file overview
/// above). See index/hnsw/report.md for the full design discussion; this
/// class is report.md §7's Phase 1 target (Region slicing + host-only
/// wiring). traverseDevice()/modifyDevice() are intentionally left at
/// IAdapter's default (throwing) implementation -- see HnswIndexDist for
/// the GPU-offload variant of traverseDevice().
///
/// Thread-safety: mirrors StressIndex's contract -- OpScheduler may call
/// traverseHost()/modifyHost() concurrently from multiple worker threads.
/// hnswlib's HierarchicalNSW already handles its own internal concurrency
/// (per-node link_list_locks_/label_lookup_lock, see thirdparty/hnswlib/
/// hnswlib/hnswalg.h); mutex_ below only serializes HnswIndex's own calls
/// into the engine (coarser than hnswlib's own fine-grained locks, but
/// simple and correct for a first cut -- revisit if profiling shows
/// contention). A subclass may reuse mutex_ for its own state too (e.g.
/// HnswIndexAnchorEntry's entry-point cache).
class HnswIndex : public IAdapter {
 public:
	HnswIndex(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
						std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction);
	~HnswIndex() override;

	HnswIndex(const HnswIndex&) = delete;
	HnswIndex& operator=(const HnswIndex&) = delete;

	/// Wraps hnswlib's own searchKnnCloserFirst() wholesale (closer-first, so
	/// no manual reversal is needed to match Neighbor's expected ordering --
	/// see test/stress/stress_index.cpp's ScanOne() for the same convention).
	///
	/// TraverseResult::touched is approximated as only the top-k results'
	/// Regions (report.md §7.2 decision (a)) -- hnswlib doesn't expose the
	/// full visited-node set through its public API without a source patch,
	/// which this class deliberately avoids (see file overview above).
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override;

	/// Insert: addPoint(), then ModifyResult::modified is the new node's own
	/// Region plus its immediate level-0 neighbors' Regions (read back via
	/// get_linklist0() right after the call) -- a conservative over-approximation
	/// (mutuallyConnectNewElement() may have also touched some of *those*
	/// neighbors' own neighbors while rebalancing, see thirdparty/hnswlib/
	/// hnswlib/hnswalg.h:554-627), but matches the level of conservatism
	/// report.md §7.2 already called for.
	/// Delete: markDelete() -- no graph rewiring, so `modified` is empty.
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override;

	// traverseDevice()/modifyDevice() intentionally not overridden here --
	// see HnswIndexDist for the GPU-offload variant of traverseDevice();
	// modifyDevice() stays permanently at IAdapter's default (report.md §3
	// decision 4 / §7.1/§7.4: Insert is Host-only by design, not a gap).

	IRegion* resolveRegion(RegionId id) override;
	std::vector<RegionId> allRegions() const override;

	/// Bulk-builds directly against hnswlib's own addPoint() -- see
	/// IAdapter::build()'s doc comment for the general contract (entirely
	/// this adapter's own responsibility, caller re-registers Regions
	/// afterward via registerAllRegions()). TEMPORARY implementation: goes
	/// straight through hnswlib's public addPoint() API one vector at a time
	/// (matching the pattern thirdparty/hnswlib's own examples/tests use for
	/// bulk load), not yet routed through modifyHost()'s Insert path --
	/// revisit once §6 (Traverse/Modify split, still not done -- see
	/// modifyHost()'s own comment) makes that reuse cheap.
	void build(const VectorBatchView& dataset) override;

	/// Registers every partition slice with `controller` as a Region --
	/// mirrors StressIndex::registerAllRegions(); must be called once before
	/// any search()/insert()/remove() goes through `controller`.
	void registerAllRegions(Controller& controller);

	/// hnswlib's own saveIndex()/loadIndex() (thirdparty/hnswlib/hnswlib/
	/// hnswalg.h), used as-is. loadFrom() additionally rebuilds regions_
	/// against the reloaded engine's (freshly realloc'd) data_level0_memory_
	/// pointer -- required by IAdapter::loadFrom()'s post-condition
	/// (resolveRegion()/allRegions() must report the same Regions after
	/// load) and by the fact hnswlib's own loadIndex() calls clear() +
	/// malloc()s a brand new block, invalidating every HnswRegion's cached
	/// pointer from before the call. Validates dim/capacity/vectors_per_region
	/// match this instance's own construction parameters first (throws
	/// std::invalid_argument otherwise), same convention as
	/// StressIndex::loadFrom().
	void exportTo(const std::string& path) const override;
	void loadFrom(const std::string& path) override;

	std::size_t liveCount() const;
	std::uint32_t dim() const { return dim_; }
	VectorDType dtype() const { return dtype_; }
	DistanceMetric metric() const { return metric_; }
	std::size_t capacity() const { return capacity_; }
	std::size_t vectorsPerRegion() const { return vectors_per_region_; }

 protected:
	/// Strategy hook: which internal id hnswlib's level-0 search should start
	/// walking from for one TraverseRequest. Default (this class's own
	/// behavior) always returns hnswlib's own global entry point --
	/// HnswIndexAnchorEntry overrides this to consult a cache first.
	virtual std::uint32_t resolveEntryPoint(const TraverseRequest& request) const;

	// engine_'s definition lives only in hnsw_index.cpp (see HnswEngine's
	// forward declaration above) so hnswlib stays a PRIVATE dependency of
	// this whole directory's public headers. Every one of the thin,
	// non-virtual forwarding helpers below exists so a subclass (in its own
	// .cpp, which does *not* see HnswEngine's definition either) can still
	// reach a specific piece of it without that definition being exposed.
	// HnswIndexDist's copied search loop (see its own file) is what needs
	// the level0-graph-shaped ones below; HnswIndexAnchorEntry only needs
	// engineGlobalEntryPoint().
	std::uint32_t engineGlobalEntryPoint() const;
	const void* engineDataPointerFor(std::uint32_t internal_id) const;
	std::vector<std::uint32_t> engineLevel0Neighbors(std::uint32_t internal_id) const;
	bool engineIsMarkedDeleted(std::uint32_t internal_id) const;
	VectorId engineExternalLabel(std::uint32_t internal_id) const;

	/// Reverse of engineExternalLabel(): external VectorId -> internal id, or
	/// nullopt if `external_id` isn't (or is no longer) in this index. Exposed
	/// for HnswIndexAnchorEntry's entry-point cache, which needs to resolve
	/// the internal id a *completed* traversal actually landed on (see its own
	/// file for why -- an Anchor's own id is not itself guaranteed to be
	/// resolvable this way, see TraverseRequest::anchor_id's doc comment).
	std::optional<std::uint32_t> engineInternalIdFor(VectorId external_id) const;

	/// Region this internal id falls under, given vectors_per_region_ --
	/// exposed (not just used internally) so HnswIndexDist's copied search
	/// loop can map a candidate's internal id to a Region for its own
	/// residency checks (see its own file).
	RegionId RegionForInternalId(std::size_t internal_id) const;

	mutable std::mutex mutex_;  // guards engine_ calls; subclasses may reuse for their own state too

 private:
	void BuildRegions();  // (re)builds regions_ from engine_'s current dataLevel0Memory()/sizeDataPerElement()

	std::uint32_t dim_;
	VectorDType dtype_;
	DistanceMetric metric_;
	std::size_t capacity_;
	std::size_t vectors_per_region_;
	std::size_t M_;
	std::size_t ef_construction_;

	std::unique_ptr<HnswEngine> engine_;  // owns the concrete hnswlib::HierarchicalNSW<DistT>
	std::vector<std::unique_ptr<HnswRegion>> regions_;
};

}  // namespace arachne::index::hnsw
