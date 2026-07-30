#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "adapter/region.hpp"
#include "types.hpp"

namespace arachne {
class Controller;
}

namespace arachne::stress {

/// One contiguous slice of StressIndex's single big buffer -- see
/// StressIndex's own class comment for why a flat slice-per-Region scheme
/// is enough here. Adapter-internal bookkeeping only (the lease epoch): the
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

/// A brute-force, full-scan "index" standing in for a real ANNS index
/// (HNSW/IVF/...), built to validate Arachne's own orchestration rather
/// than to be a good index: Arachne drives any adapter purely through
/// IndexAdapter/IRegion and never looks inside it, so a correct-but-naive
/// full scan exercises the exact same Controller machinery (routing,
/// promotion/eviction, scheduling, write-back) a real graph/cluster index
/// would, without this project needing to also get a real index's own
/// correctness right first.
///
/// One contiguous host buffer holds up to `capacity` vectors of `dim`
/// elements of `dtype`, partitioned into equal-sized slices of
/// `vectors_per_region` vectors each -- one StressRegion per slice. This is
/// StressIndex's own answer to "how does an index partition its state into
/// Regions" (still an open question for a real graph index, see todo [2]):
/// the simplest possible scheme, chosen because a flat brute-force scan
/// has no locality structure to respect in the first place. Each Region's
/// HostRegionView::subregion_bytes is set to exactly one vector's byte
/// size, so Arachne's per-Region dirty-bitmap header (gpu/dirty_header.hpp)
/// tracks dirtiness at single-vector granularity -- the natural write unit
/// once a real write kernel (stress test stage 4) starts mutating vectors
/// in place on GPU.
///
/// Thread-safety: traverseHost()/modifyHost() guard all mutable state
/// (buffer_ writes, id_to_slot_, deleted_) with mutex_, since OpScheduler
/// may run them on a worker thread concurrently with other batches (stage
/// 3's many-caller-threads stress exercises this for real; a single-caller
/// stage-1 test only ever has one batch in flight at a time, but the lock
/// costs nothing extra either way).
class StressIndex final : public IndexAdapter {
 public:
	StressIndex(std::uint32_t dim, VectorDType dtype, std::size_t capacity, std::size_t vectors_per_region);
	~StressIndex() override;

	StressIndex(const StressIndex&) = delete;
	StressIndex& operator=(const StressIndex&) = delete;

	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override;
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override;

	/// Stage 1-3 stand-in: reads the same host-mirrored buffer traverseHost()
	/// does. Safe as long as nothing writes to the *device* copy
	/// independently of Arachne's own promotion-time copyFromHost() (see
	/// Controller::make()) -- true until stage 4 adds a real write kernel,
	/// since host and device stay byte-identical otherwise. Lets a
	/// GpuOnly-routed lookup (see Controller::route()/routeSearch()) succeed
	/// instead of hitting IndexAdapter::traverseDevice()'s default throw.
	std::vector<TraverseResult> traverseDevice(const std::vector<TraverseRequest>& requests) override;

	IRegion* resolveRegion(RegionId id) override;
	std::vector<RegionId> allRegions() const override;

	/// Registers every partition slice with `controller` as a Region --
	/// must be called once before any insert()/search()/remove() goes
	/// through `controller`.
	void registerAllRegions(Controller& controller);

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
	// IndexAdapter surface.
	friend std::vector<Neighbor> BruteForceGroundTruth(const StressIndex& index, const VectorView& query,
																											std::uint32_t top_k);
};

/// Independent (of ScanOne()/traverseHost()) re-implementation of the same
/// brute-force scan, reading StressIndex's private buffer_/id_to_slot_/
/// deleted_ directly rather than calling any of its own methods. The point
/// isn't to catch a bug in StressIndex's own distance math (both
/// implementations would share that bug) -- it's to catch a bug in
/// Arachne's *orchestration* (routing a query GpuOnly vs. Hybrid, batching
/// it through OpScheduler, marshalling the TraverseResult back) by
/// comparing what Controller::search() actually returns against what's
/// verifiably in StressIndex's storage, computed without going through
/// Controller/OpScheduler/dispatch at all.
std::vector<Neighbor> BruteForceGroundTruth(const StressIndex& index, const VectorView& query, std::uint32_t top_k);

}  // namespace arachne::stress
