#pragma once

#include <atomic>
#include <optional>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "adapter/index_adapter.hpp"
#include "core/op_scheduler.hpp"
#include "core/region_manager.hpp"
#include "core/replacement_policy.hpp"
#include "core/routing_cache.hpp"
#include "gpu/device_context.hpp"
#include "gpu/device_region_pool.hpp"
#include "gpu/dirty_header.hpp"

namespace arachne {

/// Result of Controller::acquireRegion(): tells the caller where `region`'s
/// data currently lives so it can decide how to touch it -- kernel-launch
/// against `device_lease` if `on_device` is true, or handle it itself via
/// `host` (always populated; Arachne never owns host allocation, see
/// HostRegionView's doc comment) if not. This is the replacement for the
/// old, removed gpu::DeviceRegionPool::access(): a single acquire-shaped
/// entry point that answers "where is this" and, when the answer is
/// "device", hands back a Lease guaranteeing the pointer stays valid for as
/// long as the caller holds onto this result -- rather than a bare pointer
/// with no such guarantee.
///
/// Enforcement caveat (same one that applies to gpu::DeviceRegionPool::Lease
/// itself): this is a contract, not something the compiler enforces. An
/// adapter that reads `host.ptr` directly without going through this call
/// isn't stopped from doing so -- acquireRegion() exists to give a
/// cooperating caller a single place to ask "GPU or host, and is it safe
/// right now", not to make bypassing it impossible.
struct RegionAccess {
	RegionId region = 0;
	HostRegionView host;
	bool on_device = false;
	std::optional<gpu::DeviceRegionPool::Lease> device_lease;  // engaged iff on_device
};

/// Snapshot of Controller::stats(): structured, queryable state -- the
/// counterpart to the ARACHNE_LOG_DEBUG lines already threaded through
/// make()/evictAnchor()/writeBackDirtyRegions(), for callers (tests,
/// operators) that need to assert or graph on Arachne's internal state
/// rather than scrape log text. `gpu_bytes_allocated` is a live read
/// (delegates to gpu::DeviceRegionPool::bytesAllocated()); the four `_total`
/// fields are monotonically increasing counters for the Controller's whole
/// lifetime, not point-in-time state -- e.g. `regions_evicted_total` counts every Region
/// ever reclaimed, not how many are currently evicted (there's no such
/// thing as "currently evicted": an evicted Region is just back to
/// host-only, indistinguishable from one never promoted).
struct ControllerStats {
	std::size_t gpu_bytes_allocated = 0;
	std::uint64_t regions_promoted_total = 0;       // make() calls that newly acquired a lease
	std::uint64_t regions_evicted_total = 0;        // Regions reclaimed via evictAnchor() (any anchor)
	std::uint64_t regions_written_back_total = 0;   // of those, how many actually had data copied back
	std::uint64_t anchor_evictions_total = 0;       // evictAnchor() calls that reclaimed >=1 Region
	std::uint64_t compactions_total = 0;            // make() falling back to gpu::DeviceRegionPool::compact()
};

/// Arachne's core management controller: the index-agnostic control plane that
/// decides where SEARCH/INSERT/DELETE run and how GPU residency/write
/// authority is managed, per the Quick Summary. This is the class meant to
/// carry Arachne's actual design as it gets built out; everything here is
/// implemented against IndexAdapter/IRegion and RoutingCache only, never
/// against a concrete index or a concrete Anchor storage structure.
///
/// Splits "is this query close to something we've seen" (RoutingCache,
/// injected -- pure identity/routing signal, pluggable) from "which Regions
/// has that Anchor promoted, and what does each Region's own
/// host/device/lease state look like" (RegionManager, owned directly --
/// Controller's own policy-state, not pluggable). Controller is the only
/// thing that talks to both.
class Controller {
 public:
	// `replacement_policy` defaults to FifoReplacementPolicy when left null,
	// mirroring OpScheduler's own SchedulingPolicy default (see
	// core/op_scheduler.hpp) -- see ReplacementPolicy's doc comment.
	// `gpu_data_budget_bytes`/`gpu_metadata_budget_bytes` become the
	// DeviceContext this Controller owns' budgetBytes() for each MemoryKind
	// -- the ceiling make()'s gpu::DeviceRegionPool::tryAllocate() calls
	// enforce (see its own doc comment). Defaulted to the same
	// kDefaultDataPoolBytes/kDefaultMetadataPoolBytes DeviceContext itself
	// defaults to; overriding them is mainly useful for tests that want to
	// exercise capacity-exhaustion/eviction without allocating gigabytes of
	// real Region data first.
	Controller(IndexAdapter& adapter, RoutingCache& routing_cache,
			 SchedulingConfig scheduling_config = {},
			 std::unique_ptr<ReplacementPolicy> replacement_policy = nullptr,
			 std::size_t gpu_data_budget_bytes = gpu::kDefaultDataPoolBytes,
			 std::size_t gpu_metadata_budget_bytes = gpu::kDefaultMetadataPoolBytes);

	SearchResult search(const Query& query);

	/// Fails (InsertResult::ok = false, no adapter/RoutingCache/RegionManager
	/// call made at all) if record.id is already live -- either a prior
	/// insert() for the same id already succeeded and it hasn't been
	/// remove()'d since, or another thread's insert() for the same id is
	/// concurrently in flight right now. An index's own id space assumes
	/// uniqueness (e.g. StressIndex's id_to_slot_ map, or hnswlib's
	/// label_lookup_); Arachne enforces that at this boundary rather than
	/// letting a duplicate reach the adapter and silently corrupt whatever
	/// id->data mapping it keeps. remove()'ing an id frees it for reuse.
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

	/// Adapter opt-in (see RegionManager::registerRegion()'s doc comment):
	/// declares `id` promotion/eviction-eligible and records where its data
	/// lives in host memory. An index calls this once it decides a piece of
	/// its own state should be a candidate for GPU residency -- before that
	/// call, Arachne has no knowledge of `id` at all, and make() will refuse
	/// to promote any Anchor onto it.
	void registerRegion(RegionId id, HostRegionView host);

	/// Resolves where `region`'s data currently lives -- see RegionAccess's
	/// doc comment for how a caller is meant to use the result. Throws
	/// std::invalid_argument if `region` was never registered.
	RegionAccess acquireRegion(RegionId region);

	/// See ControllerStats' own doc comment. Cheap: the four counters are
	/// independent atomics read with relaxed ordering (nothing here needs to
	/// be seen-together-atomically with anything else -- each is just a
	/// running total), and gpu_bytes_allocated delegates to
	/// gpu::DeviceRegionPool::bytesAllocated(), which takes its own lock.
	ControllerStats stats() const;

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
		RegionFootprint predicted_scope;  // derived from the matched Anchor's dependent Regions
	};
		RoutingDecision route(const Query& query);
		SearchPlan routeSearch(const Query& query);
		// `candidates` is the result of the lookup traversal insert() runs
		// first (see its doc comment) -- routeInsert() folds `candidates.touched`
		// into the resulting ModifyRequest::scope and moves (not copies)
		// `candidates.hint` into ModifyRequest::hint, since `candidates` isn't
		// needed for anything else afterward.
		InsertPlan routeInsert(const Record& record, TraverseResult candidates);
		RemovePlan routeRemove(VectorId id);

	TraverseResult dispatch(const TraverseRequest& request);
	ModifyResult dispatch(const ModifyRequest& request);

	SearchResult commitSearch(const SearchPlan& plan, const TraverseResult& result,
													 bool final_was_hybrid);
	InsertResult commitInsert(const InsertPlan& plan, const ModifyResult& result);
	DeleteResult commitRemove(const RemovePlan& plan, const ModifyResult& result);

	// Workload drift (design point 2 trigger).
	void recordTraversalForDrift(bool touched_host);

	// Anchor-centric Promotion (design point 4): grants `anchor_id` a
	// dependency on every region in `footprint`. 1) registers `anchor_id` in
	// routing_cache_ under `anchor_vector` so future queries route to it, 2)
	// calls make() per region -- which, the first time any Anchor depends on
	// that region, acquires a write lease for it and allocates its GPU
	// memory, then *enqueues* (does not yet wait for) the host-to-device copy
	// of its data, appending onto the `pending`/`zero_headers` batch shared
	// across this whole call -- and 3) if make() reports
	// MakeResult::OutOfCapacity for a region, keeps asking
	// replacement_policy_ for the next Anchor to reclaim (excluding
	// `anchor_id` itself) and evicting it via evictAnchor(), retrying make()
	// after each one -- not just once, since a single victim's Regions may
	// be smaller than what's needed here. Gives up on a region (leaving it
	// unpromoted for this call) once there's no eviction candidate left, or
	// once make() reports MakeResult::NotEligible (a reason no amount of
	// eviction can fix, e.g. an unregistered region or an adapter that
	// refuses to lease it) -- a future call can try again for the capacity
	// case. Once every region in `footprint` has been attempted, flushes the
	// whole batch in a single gpu::DeviceRegionPool::flush() call -- one
	// scatter-gather round trip for the entire Anchor's footprint, mirroring
	// writeBackDirtyRegions()'s own batched-gather shape for the opposite
	// (device-to-host) direction, rather than one acquire+copy+sync per
	// Region.
	void promoteAnchor(VectorId anchor_id, const VectorView& anchor_vector,
									const RegionFootprint& footprint);

	// Reclaims every Region dependency currently held by `anchor_id`: for
	// each Region that consequently has zero remaining dependents (see
	// RegionManager::forget()'s doc comment -- Regions still depended on by
	// some other Anchor are left alone, since they're still promoted for a
	// reason), releases the write lease. The GPU-resident ones among them
	// (region.device.valid()) are then batch-written-back in one call to
	// writeBackDirtyRegions() below -- not one write-back per Region --
	// before each is freed. Then clears every Region's residency back to
	// host-only. Notifies replacement_policy_ so it stops tracking
	// `anchor_id`. The mechanism promoteAnchor() calls through
	// replacement_policy_->selectEvictionCandidate() when it needs room, and
	// the one verify() below uses on a mismatch.
	void evictAnchor(VectorId anchor_id);

	// Batched write-back for every Region in `regions` being reclaimed
	// (evictAnchor() above calls this once per evictAnchor() call, across
	// every orphaned Region that call produced -- not once per Region), each
	// still GPU-resident (region.device.valid()) and not yet freed. Arachne
	// itself never sets a dirty bit -- that's the adapter/kernel's job (it
	// atomicOr's into the per-Region dirty-bitmap header prepended to the
	// Region's device data, see gpu/dirty_header.hpp and make()'s doc
	// comment) -- so this only ever *reads* those headers back and decides
	// which Regions' data is worth copying back. Two phases, each one
	// batched gather (many Regions' device-to-host copies enqueued via
	// gpu::DeviceRegionPool::enqueueCopyToHost() on one stream, then a
	// single gpu::DeviceRegionPool::flush()) rather than one
	// acquire+copy+sync per Region:
	//  1) Gather every Region's dirty-bitmap header (skipping Regions with
	//     subregion_bytes == 0 -- see HostRegionView::subregion_bytes --
	//     which have no header at all, since there's nothing to gather for
	//     them).
	//  2) Gather the actual data for every Region that needs writing back:
	//     confirmed dirty (some header word was nonzero) or, for a Region
	//     with no header to check, conservatively assumed dirty -- Arachne
	//     has no way to distinguish dirty from clean without one, so it
	//     always writes back rather than risk silently dropping a write.
	//     Skips Regions confirmed clean, since their host copy is already
	//     authoritative and copying it back would be wasted device-to-host
	//     traffic.
	// Granularity note: phase 2 reads "any header bit set" per Region and,
	// if so, copies back that Region's *entire* data -- it does not (yet)
	// copy back just the dirty subregions individually, which would need
	// per-subregion offset/length copies instead of one contiguous gather
	// entry per Region. That refinement is future work.
	void writeBackDirtyRegions(const std::vector<Region>& regions);

	// Selective verification (design point 3). Not yet wired into search().
	// On mismatch, reclaims every Region dependency on `anchor_id` (via
	// evictAnchor()) since the regions it currently points at no longer
	// represent its locality.
	void verify(const Query& query, VectorId anchor_id, const TraverseResult& gpu_only_result);

	// Outcome of make() below, distinguishing two very different reasons a
	// Region can fail to become available for `anchor_id`: promoteAnchor()
	// only retries-via-eviction on OutOfCapacity (evicting something might
	// actually help); it gives up immediately on NotEligible (evicting
	// anything changes nothing -- the region itself, or the adapter's
	// willingness to resolve/lease it, is the problem).
	enum class MakeResult {
		Promoted,       // anchor_id now depends on region (already did, or just started).
		NotEligible,    // unregistered, adapter can't resolve/lease it -- not retryable.
		OutOfCapacity,  // registered and lease-eligible, but no room on GPU right now -- retryable.
	};

	// Capacity-aware allocation with a compaction fallback, used by make()
	// below in place of a bare gpu::DeviceRegionPool::tryAllocate() call.
	// Tries tryAllocate() first, exactly like before compaction was wired in
	// here. If that fails *and* gpu::DeviceRegionPool::hasCapacity() says
	// `bytes` should fit under the current budget anyway, this Controller
	// concludes the live bytes are fragmented (not genuinely over budget --
	// see gpu::DeviceRegionPool::compact()'s own doc comment for why
	// Controller, not DeviceRegionPool, is the one that decides when to
	// compact) and runs gpu::DeviceRegionPool::compact(MemoryKind::Data)
	// once, then retries the allocation exactly once more. Returns nullopt
	// (same as tryAllocate()) if it still doesn't fit either way -- the
	// caller (make()) treats that identically to a plain tryAllocate()
	// failure.
	std::optional<gpu::DeviceRegionHandle> allocateWithCompaction(std::size_t bytes);

	// Region promotion (design point 4): if `region` is not yet registered
	// (see registerRegion()), fails immediately -- an unregistered region was
	// never opted into promotion/eviction. If it's registered but not yet
	// promoted by anyone (region_manager_'s lease is invalid), acquires a
	// write lease for it and allocates device memory sized to
	// region_manager_.regionOf(region).host.bytes plus, if host.subregion_bytes
	// is nonzero, gpu::DirtyHeaderBytes(host.bytes, host.subregion_bytes)
	// (see gpu/dirty_header.hpp) prepended for the per-subregion dirty bitmap
	// a write kernel is expected to atomicOr into. Does not copy anything
	// synchronously: the header's zero-fill (a scratch buffer appended to
	// `zero_headers` so it outlives this call -- freshly allocated device
	// memory is not guaranteed to already be zero-filled) and the Region's
	// host data are each *enqueued* via device_region_pool_.enqueueCopyFromHost()
	// onto `pending`, which the caller (promoteAnchor()) is responsible for
	// flushing exactly once after every Region in the Anchor's footprint has
	// been attempted -- see promoteAnchor()'s own doc comment. Only once the
	// lease and the GPU allocation both succeed are the results recorded via
	// region_manager_.setLease()/setDevice() -- if the allocation can't be
	// made (see allocateWithCompaction() above), the just-acquired lease is
	// released again and nothing is recorded, so a failed attempt here never
	// leaves partial state behind. If `region` is already promoted by some
	// other Anchor, `anchor_id` simply becomes an additional dependent of the
	// existing lease -- Regions have exactly one lease (and one GPU
	// allocation) shared by every dependent, not one independent copy per
	// Anchor. Either way on success, records the dependency via
	// region_manager_.addDependency() and notifies replacement_policy_. See
	// MakeResult above for what a non-Promoted return means and who's
	// expected to react to it.
	MakeResult make(VectorId anchor_id, RegionId region, std::vector<gpu::DeviceRegionPool::Lease>& pending,
								 std::vector<std::vector<std::byte>>& zero_headers);

	IndexAdapter& adapter_;
	RoutingCache& routing_cache_;
	OpScheduler scheduler_;
	RegionManager region_manager_;
	// Strategy (design point 4): see ReplacementPolicy's doc comment.
	std::unique_ptr<ReplacementPolicy> replacement_policy_;
	// GPU residency accounting (design point 4): Arachne-owned, not the
	// adapter's -- see gpu/device_context.hpp and gpu/device_region_pool.hpp.
	// make() allocates through device_region_pool_ (capacity-aware, via
	// tryAllocate()) and evictAnchor()/writeBackDirtyRegions() free through
	// it -- see their doc comments.
	gpu::DeviceContext device_;
	gpu::DeviceRegionPool device_region_pool_;
	VectorId next_anchor_id_ = 1;
	std::uint64_t drift_window_host_ = 0;
	std::uint64_t drift_window_total_ = 0;

	// ControllerStats' backing counters -- see stats() and ControllerStats'
	// own doc comment. Independent atomics rather than a mutex-guarded
	// struct: each is bumped from a different, already-locked-elsewhere call
	// site (make(), evictAnchor(), writeBackDirtyRegions()) and read
	// independently by stats(), so there's no cross-field invariant that
	// would need them updated as one atomic unit.
	std::atomic<std::uint64_t> stat_regions_promoted_{0};
	std::atomic<std::uint64_t> stat_regions_evicted_{0};
	std::atomic<std::uint64_t> stat_regions_written_back_{0};
	std::atomic<std::uint64_t> stat_anchor_evictions_{0};
	std::atomic<std::uint64_t> stat_compactions_total_{0};

	// Which VectorIds insert() currently considers live -- see insert()'s
	// own doc comment. A plain mutex-guarded set, independent of
	// region_manager_: an id can be live (successfully handed to the
	// adapter's modifyHost()/modifyDevice()) without ever having a promoted
	// Region (promotion can fail or simply not be attempted), so
	// region_manager_.regionsOf(id) being empty is not a reliable "was this
	// id ever inserted" signal.
	std::mutex live_ids_mutex_;
	std::unordered_set<VectorId> live_ids_;
};

}  // namespace arachne
