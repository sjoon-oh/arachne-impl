#pragma once

// StressIndex: a brute-force, full-scan IAdapter implementation standing in
// for a real ANNS index (HNSW/IVF/...) in the gtest stress stages
// (unittest/stress/stress_index_test.cpp and its stage2/stage3 siblings) and
// in test/bin/full_suite_app.cpp's standalone workload runner. Its job isn't
// to be a good index -- Arachne drives any adapter purely through
// IAdapter/IRegion and never looks inside it, so a correct-but-naive full
// scan exercises exactly the same Controller machinery (routing,
// promotion/eviction, scheduling, write-back) a real graph/cluster index
// would, without this project needing a real index's own search-quality
// correctness first.
//
// Host buffer / Region layout:
// One contiguous std::vector<std::byte> buffer_ holds up to `capacity`
// vectors of `dim` elements of `dtype`, as a flat array of fixed-size vector
// slots. buffer_ is partitioned into equal-sized slices of
// `vectors_per_region` vectors each; one StressRegion (an IRegion) covers
// each slice:
//
//   buffer_: [ v0 | v1 | ... | v(vpr-1) || v(vpr) | ... | v(2*vpr-1) || ... ]
//             \________ Region 1 ________/\_________ Region 2 ________/
//
// This is StressIndex's answer to "how does an index partition its state
// into Regions" -- the simplest possible scheme, chosen because a flat
// brute-force scan has no locality structure to respect in the first place.
// Each Region's HostRegionView::subregion_bytes is set to exactly one
// vector's byte size, so Arachne's per-Region dirty-bitmap header
// (gpu/dirty_header.hpp) tracks dirtiness at single-vector granularity.
// id_to_slot_ maps a live VectorId to its slot in buffer_; next_free_slot_
// only ever grows -- a deleted slot is marked in deleted_ but never
// recycled -- so bookkeeping stays simple at the cost of eventually
// exhausting `capacity` under heavy delete/reinsert churn (the stage 3
// tests size `capacity` with this in mind).
//
// BruteForceGroundTruth() (bottom of this file) is an independent
// reimplementation of the same scan, reading StressIndex's private state
// directly instead of calling any of its own methods -- see its own doc
// comment for why.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "adapter/region.hpp"
#include "types.hpp"

namespace arachne {
class Controller;
}

namespace arachne::stress {

/// One slice of StressIndex's buffer_ -- see the file-level overview above
/// for the layout. Adapter-internal bookkeeping only (the lease epoch): the
/// actual GPU allocation/copy/free and residency decision is entirely
/// Arachne's job (Controller::make()/evictAnchor()), never this class's, per
/// IRegion's contract.
class StressRegion final : public IRegion {
 public:
	StressRegion(RegionId id, void* ptr, std::size_t bytes, std::size_t subregion_bytes);

	RegionId id() const override { return id_; }
	RegionFootprint footprint() const override { return RegionFootprint{{id_}}; }
	HostRegionView hostView() const override { return host_; }
	LeaseHandle acquireWriteLease() override { return LeaseHandle{id_, ++epoch_}; }
	void releaseWriteLease(LeaseHandle) override {}
	void applyLocalModification(LeaseHandle, const ModificationDelta&) override {}
	ReconciliationReport reconcileBoundary() override { return ReconciliationReport{}; }

 private:
	RegionId id_;
	HostRegionView host_;
	std::uint64_t epoch_ = 0;
};

/// See the file-level overview above for StressIndex's role and its host
/// buffer/Region layout (still an open question for a real graph index --
/// see todo [2]).
///
/// Thread-safety: traverseHost()/modifyHost() guard all mutable state
/// (buffer_ writes, id_to_slot_, deleted_) with mutex_, since OpScheduler
/// may run them on a worker thread concurrently with other batches (stage
/// 3's many-caller-threads stress exercises this for real; the lock costs
/// nothing extra in the single-caller stages).
class StressIndex final : public IAdapter {
 public:
	StressIndex(std::uint32_t dim, VectorDType dtype, std::size_t capacity, std::size_t vectors_per_region);
	~StressIndex() override;

	StressIndex(const StressIndex&) = delete;
	StressIndex& operator=(const StressIndex&) = delete;

	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override;
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override;

	/// Stage 1-3 stand-in: reads the same host-mirrored buffer traverseHost()
	/// does. Safe because nothing writes the *device* copy independently of
	/// Controller::make()'s promotion-time copyFromHost() until stage 4 adds
	/// a real write kernel. Lets a GpuOnly-routed lookup succeed instead of
	/// hitting IAdapter::traverseDevice()'s default throw.
	std::vector<TraverseResult> traverseDevice(const std::vector<TraverseRequest>& requests) override;

	/// Same reasoning as traverseDevice() above, for Modify: a GpuOnly insert
	/// can legitimately reach here once its lookup traversal's promotion
	/// request has already been granted a Region by the time routeInsert()
	/// checks. Delegates to modifyHost() so it succeeds instead of hitting
	/// IAdapter::modifyDevice()'s default throw.
	std::vector<ModifyResult> modifyDevice(const std::vector<ModifyRequest>& requests) override;

	IRegion* resolveRegion(RegionId id) override;
	std::vector<RegionId> allRegions() const override;

	/// Registers every partition slice with `controller` as a Region --
	/// must be called once before any insert()/search()/remove() goes
	/// through `controller`.
	void registerAllRegions(Controller& controller);

	/// See IAdapter::exportTo()/loadFrom()'s own doc comment for the general
	/// contract. Format here is a flat binary dump: a header of
	/// (dim_, dtype_, capacity_, vectors_per_region_, next_free_slot_),
	/// followed by buffer_ verbatim, one byte per capacity_ slot for
	/// deleted_, then id_to_slot_ as a count followed by (id, slot) pairs.
	/// loadFrom() throws std::invalid_argument if `path`'s header doesn't
	/// match this instance's own dim_/dtype_/capacity_/vectors_per_region_ --
	/// it never resizes buffer_/regions_ to fit, since regions_ (and any
	/// Region already registered/promoted against a live Controller) are
	/// fixed at construction time.
	void exportTo(const std::string& path) const override;
	void loadFrom(const std::string& path) override;

	std::size_t liveCount() const;
	std::uint32_t dim() const { return dim_; }
	VectorDType dtype() const { return dtype_; }

 private:
	struct Candidate {
		float dist2;
		VectorId id;
		std::size_t slot;
	};

	TraverseResult ScanOne(const TraverseRequest& request) const;
	ModifyResult InsertOne(const ModifyRequest& request);
	ModifyResult DeleteOne(const ModifyRequest& request);
	RegionId RegionForSlot(std::size_t slot) const;
	float DistanceSquared(const void* query, std::size_t slot) const;
	const void* SlotPtr(std::size_t slot) const;
	void* SlotPtr(std::size_t slot);

	std::uint32_t dim_;
	VectorDType dtype_;
	std::size_t element_size_;
	std::size_t vectors_per_region_;
	std::size_t capacity_;
	std::vector<std::byte> buffer_;  // capacity_ * dim_ * element_size_ bytes, one Region's worth per slice
	std::vector<std::unique_ptr<StressRegion>> regions_;

	mutable std::mutex mutex_;  // guards next_free_slot_/id_to_slot_/deleted_/buffer_ writes below
	std::size_t next_free_slot_ = 0;
	std::unordered_map<VectorId, std::size_t> id_to_slot_;
	std::vector<bool> deleted_;

	// See BruteForceGroundTruth()'s own doc comment for why this reaches
	// into private state directly rather than going through the public
	// IAdapter surface.
	friend std::vector<Neighbor> BruteForceGroundTruth(const StressIndex& index, const VectorView& query,
																											std::uint32_t top_k);
};

/// Independent (of ScanOne()/traverseHost()) re-implementation of the same
/// brute-force scan, reading StressIndex's private state directly rather
/// than calling any of its own methods. Not meant to catch a bug in
/// StressIndex's own distance math (both would share that) -- it's meant to
/// catch a bug in Arachne's *orchestration* by comparing what
/// Controller::search() returns against storage ground truth computed
/// without going through Controller/OpScheduler/dispatch at all.
std::vector<Neighbor> BruteForceGroundTruth(const StressIndex& index, const VectorView& query, std::uint32_t top_k);

}  // namespace arachne::stress
