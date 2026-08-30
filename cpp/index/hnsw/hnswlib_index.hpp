#pragma once

// HnswlibIndex: an IAdapter over thirdparty/hnswlib's HierarchicalNSW graph --
// the real, production ANN index report.md's Phase 1 (§7.2) plans to wire
// up, as opposed to test/stress/stress_index.hpp's brute-force test double.
//
// HnswlibIndex itself is CONCRETE (not abstract) -- it is, on its own, the
// plain/original hnswlib variant: always hnswlib's own global entry point +
// full upper-level descent, exactly like unmodified hnswlib (report.md §10,
// "옵션 2"). One sibling class, HnswlibIndexGpu (hnswlib_index_gpu.hpp),
// extends it with a GPU-offload traverseDevice(): per-hop candidate distance
// computation goes to a GPU kernel while control flow (candidate queue,
// visited set, stopping condition) stays on host (report.md §10, "옵션 1"),
// AND (originally a separate class, HnswIndexAnchorEntry, merged into this
// one -- see report.md §10.4) resolveEntryPoint() below is overridden there
// to opportunistically skip the upper-level descent via a cached Anchor id
// -> internal id, when one is available. Not to be confused with each
// other: the GPU-offload question (where distances get computed) and the
// entry-point question (where the walk starts) are independent, both
// answered by that one sibling class.
//
// Host buffer / Region layout (report.md §7.2, §2.1):
// HnswEngine (hnswlib_index.cpp, type-erased over the concrete
// hnswlib::HierarchicalNSW<DistT>) owns one contiguous data_level0_memory_
// block -- exactly the layout StressIndex's buffer_ mirrors, one fixed-size
// record per internal id (level-0 links + vector + label, see
// thirdparty/hnswlib/hnswlib/hnswalg.h). HnswlibIndex slices that block into
// equal id-range spans of `vectors_per_region` records each, one HnswRegion
// per span, matching StressIndex's scheme so Arachne's Region machinery is
// exercised identically. capacity_ is fixed at construction --
// hnswlib::HierarchicalNSW::resizeIndex() is never called (report.md §7.2:
// its realloc() would invalidate any host pointer already promoted to
// GPU).
//
// hnswlib itself is used entirely as-is: HnswEngine's TypedHnswEngine
// (hnswlib_index.cpp) calls only hnswlib's own public API/public members
// (searchKnnCloserFirst(), addPoint(), markDelete(), saveIndex(),
// loadIndex(), label_lookup_, ...) -- thirdparty/hnswlib is never patched or
// forked for this class. HnswlibIndexGpu is the one exception in this
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
/// array -- HnswlibIndex's Region unit. Structurally identical to
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
/// because HnswlibIndex needs the identical (VectorDType, DistanceMetric) ->
/// concrete hnswlib type matrix, just for the full dataset instead of a
/// small Anchor set. Defined in hnswlib_index.cpp (not this header) so hnswlib
/// stays a PRIVATE, impl-only dependency of this adapter -- same reasoning
/// as core/routing_cache_hnsw.hpp's own forward-declared Instance type.
class HnswEngine;

/// IAdapter over hnswlib's HierarchicalNSW graph, as-is (see file overview
/// above). See index/hnsw/report.md for the full design discussion; this
/// class is report.md §7's Phase 1 target (Region slicing + host-only
/// wiring). traverseDevice()/modifyDevice() are intentionally left at
/// IAdapter's default (throwing) implementation -- see HnswlibIndexGpu for
/// the GPU-offload variant of traverseDevice().
///
/// Thread-safety: mirrors StressIndex's contract -- OpScheduler may call
/// traverseHost()/modifyHost()/traverseDevice() concurrently from multiple
/// worker threads. None of the three take mutex_ (it exists solely for
/// build()/exportTo()/loadFrom()/liveCount() below -- lifecycle/persistence
/// calls that never run concurrently with the traverse/modify hot path, see
/// IAdapter::build()'s own doc comment on bypassing OpScheduler entirely).
/// Safety for the hot path instead comes from three layers stacked together:
///   1. hnswlib's HierarchicalNSW handles its own internal concurrency for
///      its own API calls (per-node link_list_locks_/label_lookup_lock/
///      global, see thirdparty/hnswlib/hnswlib/hnswalg.h) -- its own README
///      documents add_items as safe with other add_items, and knn_query as
///      safe with other knn_query.
///   2. The handful of places this adapter itself reads hnswlib state
///      directly, outside any hnswlib API call that would lock it
///      internally (label_lookup_ in TypedHnswEngine::internalIdFor()/
///      insertOne(), the level0 adjacency list in level0Neighbors(),
///      enterpoint_node_ in globalEntryPoint() -- all in hnswlib_index.cpp),
///      take the *same* public mutex hnswlib's own internal code takes for
///      the equivalent access, rather than a new lock of this class's own --
///      see each method's own comment.
///   3. (1)+(2) only cover what hnswlib itself documents as safe: same-kind
///      concurrency (search-with-search, insert-with-insert), never
///      insert-with-search. That last combination -- and Insert-vs-Delete,
///      which hnswlib doesn't document either way -- is OpScheduler's job,
///      not this class's: see IAdapter::requiresTraverseModifyIsolation()
///      (this class uses the default, true) and core/op_scheduler.hpp's
///      class doc comment for exactly what it guarantees.
/// A subclass may still reuse mutex_ for its own, unrelated state (e.g.
/// HnswlibIndexGpu's entry-point cache uses its own dedicated mutex instead,
/// precisely to stay independent of whatever mutex_ is or isn't used for
/// here -- see its own file).
class HnswlibIndex : public IAdapter {
 public:
	HnswlibIndex(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
						std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction);
	~HnswlibIndex() override;

	HnswlibIndex(const HnswlibIndex&) = delete;
	HnswlibIndex& operator=(const HnswlibIndex&) = delete;

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
	// see HnswlibIndexGpu for the GPU-offload variant of traverseDevice();
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

	/// hnswlib's own HierarchicalNSW ctor hardcodes ef_=10 internally, and
	/// nothing before this method existed ever called setEf() -- so every
	/// search through this adapter has always run at that fixed ef_search.
	/// Forwards to the underlying engine's setEf(); safe to call at any
	/// point after construction, takes effect on the next traverseHost()/
	/// traverseDevice() call. Not called by this class itself, so leaving
	/// it unset preserves today's exact behavior.
	void setEfSearch(std::size_t ef_search);

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
	// This adapter's own construction parameter, not a re-query of engine_ --
	// stable and known even before engine_ exists. hnswlib itself derives its
	// level-0 max degree from this as maxM0_ = M_ * 2 at construction time
	// (thirdparty/hnswlib/hnswlib/hnswalg.h) and never changes it afterward;
	// HnswlibIndexGpu's scratch-sizing (hnswlib_index_gpu.cpp) relies on that
	// fixed relationship.
	std::size_t M() const { return M_; }

 protected:
	/// Strategy hook: which internal id hnswlib's level-0 search should start
	/// walking from for one TraverseRequest. Default (this class's own
	/// behavior) always returns hnswlib's own global entry point --
	/// HnswlibIndexGpu overrides this to consult its entry-point cache first.
	virtual std::uint32_t resolveEntryPoint(const TraverseRequest& request) const;

	// engine_'s definition lives only in hnswlib_index.cpp (see HnswEngine's
	// forward declaration above) so hnswlib stays a PRIVATE dependency of
	// this whole directory's public headers. Every one of the thin,
	// non-virtual forwarding helpers below exists so a subclass (in its own
	// .cpp, which does *not* see HnswEngine's definition either) can still
	// reach a specific piece of it without that definition being exposed --
	// HnswlibIndexGpu's copied search loop (see its own file) needs the
	// level0-graph-shaped ones below plus engineGlobalEntryPoint().
	std::uint32_t engineGlobalEntryPoint() const;
	const void* engineDataPointerFor(std::uint32_t internal_id) const;
	std::vector<std::uint32_t> engineLevel0Neighbors(std::uint32_t internal_id) const;
	bool engineIsMarkedDeleted(std::uint32_t internal_id) const;
	VectorId engineExternalLabel(std::uint32_t internal_id) const;

	/// hnswlib's own already-selected distance function (HierarchicalNSW::
	/// fstdistfunc_/dist_func_param_, set once from the Space at construction
	/// -- see thirdparty/hnswlib/hnswlib/hnswalg.h's setEf()-adjacent ctor
	/// code), called directly on two raw vector-data pointers. `a`/`b` may
	/// point at host memory that isn't part of any Region (e.g. a query
	/// vector) as freely as at engineDataPointerFor()'s own return value --
	/// hnswlib's distance functions never care where a buffer came from, only
	/// its layout. Used by HnswlibIndexGpu's host-side fallback for a candidate
	/// whose Region isn't GPU-resident (hnswlib_index_gpu.cpp) -- calling
	/// hnswlib's own function here (rather than re-deriving the formula a
	/// second time) is what guarantees this matches traverseHost() exactly,
	/// including whichever SIMD tier hnswlib's own runtime CPU-capability
	/// detection selected. Pure/read-only (fstdistfunc_/dist_func_param_ are
	/// set once at construction and never mutated again), so safe to call
	/// concurrently from any number of threads with no locking at all.
	float engineHostDistance(const void* a, const void* b) const;

	/// Reverse of engineExternalLabel(): external VectorId -> internal id, or
	/// nullopt if `external_id` isn't (or is no longer) in this index. Exposed
	/// for HnswlibIndexGpu's entry-point cache, which needs to resolve the
	/// internal id a *completed* traversal actually landed on (see its own
	/// file for why -- an Anchor's own id is not itself guaranteed to be
	/// resolvable this way, see TraverseRequest::anchor_id's doc comment).
	std::optional<std::uint32_t> engineInternalIdFor(VectorId external_id) const;

	/// Region this internal id falls under, given vectors_per_region_ --
	/// exposed (not just used internally) so HnswlibIndexGpu's copied search
	/// loop can map a candidate's internal id to a Region for its own
	/// residency checks (see its own file).
	RegionId RegionForInternalId(std::size_t internal_id) const;

	// Guards only build()/exportTo()/loadFrom()/liveCount() -- see class doc
	// comment's "Thread-safety" section for why traverseHost()/modifyHost()/
	// traverseDevice() don't take this.
	mutable std::mutex mutex_;

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
