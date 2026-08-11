#pragma once

// HnswIndex: an IAdapter over thirdparty/hnswlib's HierarchicalNSW graph --
// the real, production ANN index report.md's Phase 1 (§7.2) plans to wire
// up, as opposed to test/stress/stress_index.hpp's brute-force test double.
//
// [SKELETON] This file is structure only -- see index/hnsw/report.md for
// the full design discussion. Every method whose body would need actual
// hnswlib traversal/modification logic currently throws
// std::logic_error("... not yet implemented (skeleton)"); only the
// structural parts (construction, Region slicing/registration,
// resolveRegion()/allRegions()) are real. Search this file and
// hnsw_index.cpp for "TODO(Phase" to find what's still open.
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
// Known open problem carried over from report.md §10.3, not solved by this
// skeleton: hnswlib internal ids are assigned in *insertion* order, not
// spatial order, so an id-contiguous Region does not necessarily correspond
// to a graph-local neighborhood. Left as-is per report.md's decision to
// treat this as a separate, later concern (§10.5).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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

/// IAdapter over hnswlib's HierarchicalNSW graph. See index/hnsw/report.md
/// for the full design discussion; this class is report.md §7's Phase 1
/// target (Region slicing + host-only wiring, no GPU kernel yet -- Phase 2
/// per §7.3/§9 is future work, so traverseDevice()/modifyDevice() are
/// intentionally left at IAdapter's default (throwing) implementation).
///
/// Thread-safety: mirrors StressIndex's contract -- OpScheduler may call
/// traverseHost()/modifyHost() concurrently from multiple worker threads.
/// hnswlib's HierarchicalNSW already handles its own internal concurrency
/// (per-node link_list_locks_/label_lookup_lock, see thirdparty/hnswlib/
/// hnswlib/hnswalg.h); mutex_ below only guards HnswIndex's own bookkeeping
/// around it (currently just anchor_entry_point_).
class HnswIndex final : public IAdapter {
 public:
	HnswIndex(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
						std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction);
	~HnswIndex() override;

	HnswIndex(const HnswIndex&) = delete;
	HnswIndex& operator=(const HnswIndex&) = delete;

	/// TODO(Phase 1, report.md §6): split into a Traverse call
	/// (searchBaseLayer-equivalent, read-only) and Modify call
	/// (mutuallyConnectNewElement-equivalent) against the underlying
	/// HnswEngine. Report.md's accepted first cut wraps searchKnn()/
	/// addPoint() whole rather than hnswlib's private internals, at the cost
	/// of insert() searching twice -- not yet implemented either way.
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override;
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override;

	// traverseDevice()/modifyDevice() intentionally not overridden here --
	// Phase 1 is host-only (report.md §7.2); IAdapter's default (throwing)
	// implementation applies until Phase 2 (§7.3/§9) adds a real kernel.

	IRegion* resolveRegion(RegionId id) override;
	std::vector<RegionId> allRegions() const override;

	/// Bulk-builds directly against hnswlib's own addPoint() -- see
	/// IAdapter::build()'s doc comment for the general contract (entirely
	/// this adapter's own responsibility, caller re-registers Regions
	/// afterward via registerAllRegions()). TEMPORARY implementation: goes
	/// straight through hnswlib's public addPoint() API one vector at a time
	/// (matching the pattern thirdparty/hnswlib's own examples/tests use for
	/// bulk load), not yet the "real" Traverse/Modify-split path
	/// traverseHost()/modifyHost() above still need (report.md §6) --
	/// revisit once that split exists, since at that point build() should
	/// presumably drive the same internal path modifyHost()'s Insert case
	/// does, just without Controller's per-record routing overhead.
	void build(const VectorBatchView& dataset) override;

	/// Registers every partition slice with `controller` as a Region --
	/// mirrors StressIndex::registerAllRegions(); must be called once before
	/// any search()/insert()/remove() goes through `controller`. Fully
	/// implemented (not a stub): Region slicing/registration is structural
	/// and needed for the rest of Arachne's machinery to have something to
	/// drive, independent of whether traverse/modify are wired up yet.
	void registerAllRegions(Controller& controller);

	/// TODO(export/load): needs a binary format decision for hnswlib's own
	/// state (data_level0_memory_, linkLists_, label_lookup_, enterpoint_node_/
	/// maxlevel_, ...) -- hnswlib's own saveIndex()/loadIndex()
	/// (thirdparty/hnswlib/hnswlib/hnswalg.h) is the obvious starting point
	/// but doesn't by itself satisfy IAdapter::loadFrom()'s post-condition
	/// (resolveRegion()/allRegions() must report the same Regions after
	/// load) -- regions_ still needs rebuilding against the reloaded engine's
	/// memory. Not yet implemented.
	void exportTo(const std::string& path) const override;
	void loadFrom(const std::string& path) override;

	std::size_t liveCount() const;
	std::uint32_t dim() const { return dim_; }
	VectorDType dtype() const { return dtype_; }

 private:
	RegionId RegionForInternalId(std::size_t internal_id) const;

	std::uint32_t dim_;
	VectorDType dtype_;
	DistanceMetric metric_;
	std::size_t capacity_;
	std::size_t vectors_per_region_;
	std::size_t M_;
	std::size_t ef_construction_;

	std::unique_ptr<HnswEngine> engine_;  // owns the concrete hnswlib::HierarchicalNSW<DistT>
	std::vector<std::unique_ptr<HnswRegion>> regions_;

	mutable std::mutex mutex_;  // guards anchor_entry_point_ (see its own doc comment below)

	/// TODO(report.md §10.4, entry-point reuse): Anchor id -> hnswlib
	/// internal id (tableint), so a GpuOnly-retried query can skip the
	/// upper-level descent RoutingCache already made redundant. Needs
	/// TraverseRequest (include/adapter/index_adapter.hpp) to carry an
	/// anchor_id -- a small Core change described in report.md §10.4 point 1,
	/// not yet made -- before this map can be usefully populated or read.
	std::unordered_map<VectorId, std::uint32_t> anchor_entry_point_;
};

}  // namespace arachne::index::hnsw
