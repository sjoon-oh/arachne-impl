#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "adapter/region.hpp"
#include "core/replacement_policy.hpp"
#include "core/routing_cache.hpp"
#include "gpu/device_region_pool.hpp"
#include "telemetry/instrumented_mutex.hpp"
#include "types.hpp"

namespace arachne {

// =============================================================================
// RegionManager -- GPU residency policy owner
// =============================================================================
//
// RegionManager owns every registered Region (the physical host/device
// mapping for one RegionId), the bipartite Anchor<->Region dependency graph,
// the Anchor replacement policy, and -- once start() is called -- the
// Coordinator: a single dedicated background thread that performs all actual
// GPU residency management (Promotion, Eviction, Write-back, Compaction)
// lazily and batched, off of any Controller-calling thread's critical path.
//
// This used to be split: Controller owned replacement_policy_ and ran
// make()/promoteAnchor()/evictAnchor()/writeBackDirtyRegions() synchronously,
// inline in insert()/remove(); RegionManager was pure bookkeeping. It is now
// unified here, since "which Anchor to evict, which Region to promote next,
// and when" is a single cohesive policy question. Controller is left
// responsible only for routing and scheduling (OpScheduler), never GPU
// residency policy.
//
// Synchronous vs. deferred work
// ------------------------------
// Two things stay strictly synchronous (never deferred to the Coordinator),
// because callers depend on seeing their effect immediately:
//   - The bipartite dependency graph itself (registerRegion()/
//     addDependency()/forget()/etc) -- so regionsOf()/route()'s GPU-only
//     routing decision is never stale.
//   - releaseAnchor()'s write-lease release and clearResidency(), plus its
//     replacement_policy_->onAnchorEvicted() notification -- so a Region
//     orphaned by one Anchor is immediately eligible for re-promotion under
//     a different Anchor, without waiting for the Coordinator to free its
//     old GPU allocation first.
//
// Everything else -- requestPromotion()'s enqueue, and every subsequent step
// (handing candidates to the replacement policy, deciding what to
// promote/evict, the actual GPU work) -- happens only on the Coordinator
// thread. This keeps every replacement-policy decision on one single thread,
// so a policy implementation never has to reason about concurrent callers
// touching its own ranking state -- only onAnchorEvicted() (from
// releaseAnchor()) and onAnchorTouched() (from recordTraversal()) are still
// invoked from whatever thread the underlying event happens on.
//
// RoutingCache ownership: RegionManager now owns *writing* to RoutingCache
// (Controller still only reads it, via RoutingCache::nearest() in route()).
// ensure() is called from processPromotions(), once a candidate's footprint
// actually gained at least one Region dependency; erase() is called from
// both releaseAnchor() and evictAnchorNow(), whenever an Anchor stops being
// GPU-resident for any reason (delete, verification mismatch, or plain
// capacity-driven eviction). This used to live in Controller's
// commitSearch()/commitInsert(), unconditionally -- see PromotionCandidate's
// own doc comment (replacement_policy.hpp) for the vector-lifetime and
// epoch-staleness machinery this move required.
//
// Key invariants
// --------------
//  - Epoch staleness guard: releaseAnchor() bumps anchor_epoch_[anchor_id]
//    before erasing from RoutingCache. Any PromotionCandidate for that
//    Anchor already enqueued (in pending_promotions_, or already admitted
//    into the policy) carries the old epoch and is discarded outright by
//    processPromotions() rather than acted on -- this is what stops a
//    deleted-then-VectorId-reused Anchor from reviving stale
//    dependency-graph/RoutingCache state under the new insert's identity.
//  - allocateWithCompaction() fallback chain: tryAllocate() first; if
//    genuinely over budget, give up; otherwise flush and release any Lease
//    still pinned from earlier in this same batch, run compact(), then
//    tryAllocate() once more. compact() only ever relocates *unpinned*
//    allocations, so flushing already-promoted Leases first widens the pool
//    of movable blocks instead of needlessly excluding Regions this same
//    batch is itself responsible for pinning.
//  - OutOfCapacity retry loop (processPromotions()): a make() call
//    returning OutOfCapacity is retried by evicting a victim
//    (selectNextEvictionCandidate()) and calling make() again, looping
//    until Promoted or no victim is left. Pending Leases are flushed before
//    every eviction, because evictAnchorNow()'s free() can block waiting on
//    a Lease this same pass is still holding open -- without the flush that
//    is a real deadlock (the Coordinator thread waiting on itself), not
//    just a missed optimization.
//
// Thread-safety: every bookkeeping method takes RegionManager's own lock.
// requestPromotion()/releaseAnchor()/waitIdle() are safe to call from any
// number of concurrent Controller-calling threads; the Coordinator itself
// is the only thread that ever performs actual GPU work, so that work is
// inherently serialized without needing its own separate lock.
//
// Request / Coordinator flow
// ---------------------------
//
//   Controller-calling threads                Coordinator thread (single)
//   ---------------------------               ----------------------------
//   requestPromotion()                        coordinatorLoop():
//     -> enqueue PromotionCandidate              wakes every trigger_interval,
//        onto pending_promotions_                or immediately if force-woken
//                                                 (waitIdle()/shutdown())
//   releaseAnchor()  (synchronous)                       |
//     -> forget() + lease release +                      v
//        clearResidency() + epoch++             drain pending_promotions_
//     -> pending_reclaims_.push_back()             -> replacement_policy_
//                                                        ->enqueueCandidate()
//   recordTraversal()                           drain pending_reclaims_
//     -> replacement_policy_                       -> reclaimRegions()
//          ->onAnchorTouched()                              |
//                                                            v
//                                            forced / stop / onRelocationTrigger()?
//                                                            | yes
//                                                            v
//                                            processPromotions():
//                                              while selectNextPromotionCandidate():
//                                                epoch stale? -> discard candidate
//                                                for region in footprint:
//                                                  make(region)
//                                                    -> allocateWithCompaction()
//                                                         tryAllocate()
//                                                         fail -> compact() -> tryAllocate()
//                                                  OutOfCapacity?
//                                                    -> selectNextEvictionCandidate()
//                                                    -> evictAnchorNow(victim)
//                                                    -> retry make()
//                                                any Region promoted?
//                                                  -> routing_cache_->ensure()
//                                              one flush() for the whole pass
//
// =============================================================================

// mutex_'s type is switched by ARACHNE_ENABLE_TRACING (see
// telemetry/instrumented_mutex.hpp): on, it records lock/idle-cv wait time;
// off, it's plain std::mutex. condition_variable::wait() only accepts
// std::unique_lock<std::mutex>, so the two condition variables below must
// switch to condition_variable_any whenever mutex_ isn't literally std::mutex.
#ifdef ARACHNE_ENABLE_TRACING
class RegionManagerMutex : public telemetry::InstrumentedMutex {
 public:
	RegionManagerMutex() : InstrumentedMutex("RegionManager") {}
};
using RegionManagerCondVar = std::condition_variable_any;
#else
using RegionManagerMutex = std::mutex;
using RegionManagerCondVar = std::condition_variable;
#endif

/// The single physical-mapping record for one RegionId: host location
/// (`host`), GPU location once promoted (`device`), and the current GPU
/// write authority (`lease`). There is exactly one Region per RegionId --
/// unlike the old per-Anchor Stitch, every Anchor depending on the same
/// Region observes the same host/device/lease state.
enum class RegionResidencyState {
	HostOnly,
	Promoting,
	Resident,
	Retiring,
};

struct Region {
	RegionId id = 0;
	HostRegionView host;
	gpu::DeviceRegionHandle device;
	LeaseHandle lease;
	RegionResidencyState residency_state = RegionResidencyState::HostOnly;
	std::uint64_t residency_generation = 0;
	std::size_t residency_pins = 0;
};

/// Configuration for the Coordinator thread (see start()). Promotion enqueue
/// wakes the event-driven consumer immediately; `trigger_interval` is the
/// coalescing window between the first prepared candidate and batch commit,
/// not an intake polling interval.
struct CoordinatorConfig {
	std::chrono::milliseconds trigger_interval{100};
	/// Soft per-pass movement limits. A single oversized first candidate/victim
	/// is allowed so progress is possible; subsequent work is returned to the
	/// policy for a later pass -- unconditionally, even during a waitIdle()/
	/// shutdown()-forced drain, since a candidate bumped by this limit is only
	/// ever requeued once at least one other candidate has already been
	/// admitted in the same pass, which by itself guarantees the Coordinator's
	/// per-pass retry loop keeps making progress (see buildRelocationPlan()'s
	/// own comment for the termination argument). Zero disables the
	/// corresponding limit.
	std::size_t max_promotion_bytes_per_pass = 0;
	std::size_t max_eviction_bytes_per_pass = 0;
	/// Convenience alternative to the two absolute-byte fields above,
	/// expressed as a fraction of the actual GPU data budget (e.g. 0.2 for
	/// "no single pass moves more than 20% of capacity") instead of a raw
	/// byte count. Resolved by Controller's constructor -- not read by
	/// RegionManager itself -- right after DeviceContext is constructed, so
	/// it reflects the *real* post-unit-rounding budget (gpu::DeviceContext::
	/// budgetBytes()) rather than the raw gpu_data_budget_bytes a caller
	/// requested, which can differ under AllocationPolicy::Pooled. When set,
	/// it overwrites max_promotion_bytes_per_pass/max_eviction_bytes_per_pass
	/// with the resolved byte value before RegionManager::start() is called.
	/// std::nullopt (the default) leaves the corresponding *_bytes_per_pass
	/// field exactly as given -- constructing a Controller without ever
	/// touching these two fields is unaffected either way.
	std::optional<double> max_promotion_fraction_of_budget;
	std::optional<double> max_eviction_fraction_of_budget;
	/// Minimum physical-reservation utilization required for near-fit handle
	/// reuse, expressed as an integer percentage. The default 90 rejects a
	/// 1-unit target reusing a 2-unit victim slot; 0 permits any fitting slot,
	/// and values above 100 are clamped to 100 during start().
	std::uint8_t near_fit_min_utilization_percent = 90;

	/// Anchor-group formation for eviction accounting (see region_manager.cpp's
	/// assignAnchorToGroup() and buildEvictionCandidates()). A newly-promoted
	/// Anchor joins the existing group its footprint overlaps most with if that
	/// overlap (as a fraction of the Anchor's own footprint region count) is at
	/// least this threshold *and* the group has room (see
	/// max_eviction_group_size below); otherwise it starts a new, singleton
	/// group of its own. Groups only ever grow or dissolve, never merge --
	/// keeps the bookkeeping O(1) amortized per promotion, with no risk of two
	/// large groups being unioned into one unbounded one.
	///
	/// Why this matters: `dependents_[region]` (region_manager.cpp) tracks
	/// every Anchor currently sharing a Region, but a Region is only ever
	/// reclaimable by evicting *all* of them together (see
	/// EvictionCandidate::group_members's doc comment) -- grouping
	/// substantially-overlapping Anchors together up front means a single
	/// eviction decision can actually name a set worth evicting jointly,
	/// instead of the replacement policy discovering (or failing to discover)
	/// that combination one Anchor at a time.
	double group_merge_overlap_threshold = 0.5;

	/// Hard cap on how many Anchors a single group may accumulate (checked at
	/// join time, not retroactively) -- a structural safety valve independent
	/// of whatever a plugged-in ReplacementPolicy decides is "hot": a Region
	/// this popular is left alone by group-based eviction entirely (its
	/// Anchors simply keep spilling into fresh singleton-or-small groups of
	/// their own once this group is full) rather than risking one eviction
	/// decision evicting a disruptively large number of Anchors at once.
	/// Defaults to 1 -- every Anchor gets its own singleton group, which
	/// reproduces this port's original sole-ownership-only reclaimability
	/// rule exactly (zero behavior change unless explicitly raised).
	std::size_t max_eviction_group_size = 1;
};

/// GPU residency policy owner -- see the file-level overview above for
/// RegionManager's design, invariants, and the Coordinator request-flow
/// diagram.
class RegionManager {
 public:
	/// `replacement_policy` defaults to CostAwareReplacementPolicy when left
	/// null. FIFO/LRU/LFU/Clock/2Q and external policies remain injectable.
	explicit RegionManager(std::unique_ptr<ReplacementPolicy> replacement_policy = nullptr);
	~RegionManager();

	RegionManager(const RegionManager&) = delete;
	RegionManager& operator=(const RegionManager&) = delete;

	/// Registers `id` as promotion/eviction-eligible and records `host` as
	/// where its data currently lives. No-op if `id` is already registered --
	/// re-registering does not refresh `host`; a moved host allocation is out
	/// of scope for this skeleton (see IRegion::hostView()'s doc comment).
	void registerRegion(RegionId id, HostRegionView host);

	bool isRegistered(RegionId id) const;

	/// Snapshot of `id`'s current Region record. Throws std::invalid_argument
	/// if `id` was never registered.
	Region regionOf(RegionId id) const;

	/// RegionIds `anchor_id` currently depends on (empty if none).
	std::vector<RegionId> regionsOf(VectorId anchor_id) const;

	/// Returns an all-or-nothing routing snapshot for an Anchor. Empty means
	/// that at least one dependency is not currently publishable as Resident.
	std::vector<RegionResidencyHint> residencyHints(VectorId anchor_id) const;

	/// Atomically validates and pins every hinted Region. A null result means
	/// the routing hint became stale; callers should use the Hybrid path.
	/// The returned opaque guard unpins all Regions when its last copy dies.
	std::shared_ptr<void> tryPinResidency(const std::vector<RegionResidencyHint>& hints);

	/// Records `anchor_id` as a dependent of `region_id` (idempotent). Returns
	/// false without recording anything if `region_id` was never registered --
	/// callers (make()) are expected to check this before attempting to
	/// promote a region nobody opted in.
	bool addDependency(VectorId anchor_id, RegionId region_id);

	/// Drops `anchor_id`'s dependency on `region_id`. Returns true exactly
	/// when that was the *last* Anchor depending on it -- the caller's signal
	/// to reclaim the lease/device residency via clearResidency() below.
	/// Returns false (no-op) if there was no such dependency, or others remain.
	bool removeDependency(VectorId anchor_id, RegionId region_id);

	/// Drops every dependency `anchor_id` has, returning the RegionIds that
	/// consequently dropped to zero dependents -- the set the caller should
	/// reclaim (release lease, free device memory, clearResidency()). Regions
	/// still depended on by another Anchor are excluded, per removeDependency().
	std::vector<RegionId> forget(VectorId anchor_id);

	/// Updates region `id`'s lease field, e.g. right after make() calls
	/// IRegion::acquireWriteLease() for it. No-op if `id` isn't registered.
	void setLease(RegionId id, LeaseHandle lease);

	/// Updates region `id`'s device field, e.g. right after make() allocates
	/// GPU memory for it through gpu::DeviceRegionPool. No-op if `id` isn't
	/// registered.
	void setDevice(RegionId id, gpu::DeviceRegionHandle device);

	/// Resets region `id` back to host-only (invalid lease and device),
	/// leaving it registered (host mapping untouched). No-op if `id` isn't
	/// registered.
	void clearResidency(RegionId id);

	/// Starts the Coordinator thread, mirroring OpScheduler::start()'s
	/// lifecycle shape. `adapter`/`device_region_pool`/`routing_cache` are
	/// owned outside RegionManager (by Controller) and must outlive it. Must
	/// be called before requestPromotion()/releaseAnchor() are used. Throws
	/// std::logic_error if already started.
	void start(IAdapter& adapter, gpu::DeviceRegionPool& device_region_pool, RoutingCache& routing_cache,
						 CoordinatorConfig config = {});

	/// Stops the Coordinator, draining whatever it's currently processing (or
	/// picks up in one last pass) before returning -- mirrors
	/// OpScheduler::shutdown(). Safe to call if never started, or already
	/// shut down (no-op).
	void shutdown();

	/// Pure multiple-producer enqueue onto RegionManager's own intake queue --
	/// returns immediately; only the Coordinator (the queue's single
	/// consumer) later hands the candidate to replacement_policy_. `vector`
	/// is copied immediately (see PromotionCandidate's doc comment) and
	/// stamped with `anchor_id`'s current epoch, so a candidate outlived by a
	/// releaseAnchor() call is discarded rather than promoted/registered.
	void requestPromotion(VectorId anchor_id, RegionFootprint footprint, VectorView vector = {});

	/// Synchronously drops every Region dependency `anchor_id` has, releases
	/// the write lease on each Region left with zero dependents, clears its
	/// residency, and erases `anchor_id` from RoutingCache -- so a later
	/// requestPromotion() for a different Anchor wanting the same Region is
	/// never blocked behind this Region's still-pending GPU reclaim. Only the
	/// write-back-if-dirty + free of the now-orphaned device memory is
	/// deferred to the Coordinator. Also bumps `anchor_id`'s epoch (see the
	/// epoch staleness invariant above) so any already-enqueued
	/// PromotionCandidate for it is discarded rather than acted on.
	void releaseAnchor(VectorId anchor_id);

	/// True if `anchor_id` has ever been assigned to something -- either it
	/// currently depends on at least one Region (dependencies_), or it was
	/// released at some point in its lifetime (anchor_epoch_ entries are
	/// never removed -- see that member's own doc comment). Exists purely
	/// for Controller::MintAnchorId()'s wraparound-collision guard: once
	/// (astronomically unlikely to ever happen -- see MintAnchorId()'s own
	/// doc comment) `next_anchor_id_` has cycled through its entire 64-bit
	/// range, a freshly minted candidate id needs to be checked against
	/// whatever's still using an id from the previous cycle before being
	/// handed out again. The one gap this doesn't close: an Anchor that's
	/// been minted and enqueued as a PromotionCandidate but not yet promoted
	/// into a Region dependency -- covering that would mean plumbing an
	/// equivalent query through every ReplacementPolicy implementation for a
	/// scenario that additionally requires landing exactly on that one
	/// pending id before it resolves, judged not worth the coupling (see
	/// MintAnchorId()'s own doc comment for the full reasoning).
	bool isKnownAnchor(VectorId anchor_id) const;

	/// Reports that a traversal actually accessed every Region in `touched`
	/// -- forwarded as onAnchorTouched() to every Anchor currently depending
	/// on one of those Regions, deduplicated. Best-effort hotness signal, not
	/// bookkeeping (never throws); synchronous and cheap, unlike
	/// requestPromotion()/releaseAnchor() there is no GPU work to defer here.
	void recordTraversal(const RegionFootprint& touched);

	/// Blocks until the Coordinator has fully drained everything requested
	/// via requestPromotion()/releaseAnchor() so far (including anything
	/// requested concurrently while this call waits) -- for callers (tests,
	/// an operator) needing a guarantee residency reflects every request
	/// made, not just that it was accepted. Forces an immediate wake rather
	/// than waiting out trigger_interval. Not needed on the normal async path.
	void waitIdle();

	/// Snapshot of RegionManager's own internal counters, for callers (tests,
	/// operators) that need to assert or graph on GPU residency state rather
	/// than scrape log text. `gpu_bytes_allocated` is a live read (delegates
	/// to gpu::DeviceRegionPool::bytesAllocated()); the rest are monotonically
	/// increasing counters for this RegionManager's whole lifetime.
	struct Stats {
		std::size_t gpu_bytes_allocated = 0;
		std::uint64_t regions_promoted_total = 0;
		std::uint64_t regions_evicted_total = 0;
		std::uint64_t regions_written_back_total = 0;
		std::uint64_t anchor_evictions_total = 0;
		std::uint64_t compactions_total = 0;
		std::uint64_t relocation_batches_total = 0;
		std::uint64_t candidates_requeued_total = 0;
		std::uint64_t near_fit_reuses_total = 0;
		/// Promotion candidates a ReplacementPolicy::evaluateBatchAdmission()
		/// permanently dropped via BatchAdmissionDecision::Reject (see that
		/// enum's own doc comment for how this differs from a requeue) --
		/// tracked centrally here so any policy gets this observability for
		/// free, without needing its own counter.
		std::uint64_t candidates_rejected_total = 0;
	};
	Stats stats() const;

 private:
	// Outcome of make() below, distinguishing two reasons a Region can fail
	// to become available: processPromotions() only retries-via-eviction on
	// OutOfCapacity (evicting something might help); it gives up immediately
	// on NotEligible (the region/adapter itself is the problem, not capacity).
	enum class MakeResult {
		Promoted,       // anchor_id now depends on region (already did, or just started).
		NotEligible,    // unregistered, adapter can't resolve/lease it -- not retryable.
		OutOfCapacity,  // registered and lease-eligible, but no room on GPU right now -- retryable.
		Deferred,       // another transition currently owns this Region -- retry in a later batch.
	};

	// Capacity-aware allocation with a compaction fallback (tryAllocate ->
	// compact -> retry), used by make() in place of a bare
	// gpu::DeviceRegionPool::tryAllocate() call. See the "allocateWithCompaction()
	// fallback chain" invariant in the file-level overview above for why
	// `pending` must be flushed before compact() runs.
	std::optional<gpu::DeviceRegionHandle> allocateWithCompaction(
			std::size_t bytes, gpu::DeviceRegionPool::TransferBatch& pending);
	struct PendingPromotionCommit {
		VectorId anchor_id = 0;
		RegionId region_id = 0;
		gpu::DeviceRegionHandle device;
		LeaseHandle lease;
		std::uint64_t residency_generation = 0;
		std::uint64_t anchor_epoch = 0;
	};
	using ReusableAllocations = std::unordered_map<RegionId, gpu::DeviceRegionHandle>;

	struct PlannedPromotion {
		PromotionCandidate candidate;
		AdmissionContext admission;
	};
	struct RelocationPlan {
		std::uint64_t batch_sequence = 0;
		std::vector<PlannedPromotion> promotions;
		std::vector<VectorId> evictions;
		std::size_t required_incremental_bytes = 0;
		std::size_t immediately_reclaimable_bytes = 0;
	};
	struct PromotionStorageRequest {
		RegionId region_id = 0;
		std::size_t logical_bytes = 0;
		std::size_t reserved_bytes = 0;
	};

	struct ResidencyPinBatch {
		RegionManager* owner = nullptr;
		std::vector<RegionResidencyHint> hints;
		~ResidencyPinBatch();
	};
	void unpinResidency(const std::vector<RegionResidencyHint>& hints);

	// Region promotion (design point 4): acquires a write lease and enqueues
	// (does not wait for) the host-to-device copy of `region`'s data if it
	// isn't already promoted, then records `anchor_id` as a dependent. See
	// MakeResult above for what a non-Promoted return means. Does not touch
	// replacement_policy_ or RoutingCache -- selectNextPromotionCandidate()
	// and processPromotions() own those, respectively (see their own doc
	// comments).
	MakeResult make(VectorId anchor_id, std::uint64_t anchor_epoch, RegionId region,
			gpu::DeviceRegionPool::TransferBatch& pending,
			std::vector<PendingPromotionCommit>& commits, ReusableAllocations& reusable);

	// `eviction_candidates_cache` is populated lazily (only once eviction info
	// is actually needed -- see AdmissionContext::eviction_candidates' own
	// doc comment) and reused across every candidate examined within the same
	// buildRelocationPlan() pass: residency never changes mid-pass (nothing
	// in that loop promotes/evicts anything -- only processRelocationBatch()
	// does, afterward), so recomputing buildEvictionCandidates() more than
	// once per pass would only ever reproduce the exact same result.
	AdmissionContext buildAdmissionContext(
			const PromotionCandidate& candidate, std::optional<std::vector<EvictionCandidate>>& eviction_candidates_cache) const;
	std::vector<EvictionCandidate> buildEvictionCandidates() const;
	std::size_t reservedRegionBytes(const Region& region) const;
	std::size_t promotionBytes(const std::vector<PlannedPromotion>& promotions) const;
	std::size_t projectedReclaimableBytes(const std::vector<VectorId>& victims,
			bool require_unpinned) const;
	std::vector<PromotionStorageRequest> buildPromotionStorageRequests(
			const std::vector<PlannedPromotion>& promotions) const;

	// `anchor_id`'s current epoch, 0 if it has never had one (releaseAnchor()
	// is the only thing that bumps it). Caller must already hold mutex_.
	std::uint64_t currentEpochLocked(VectorId anchor_id) const;

	// Drives one Coordinator pass's worth of promotion work: repeatedly pulls
	// from replacement_policy_->selectNextPromotionCandidate() until nullopt,
	// discards epoch-stale candidates, retries OutOfCapacity results via
	// eviction (see the OutOfCapacity retry loop invariant above), then
	// registers each Anchor in RoutingCache iff it gained at least one Region
	// dependency. Flushes every host-to-device copy from the whole pass once.
	std::optional<RelocationPlan> buildRelocationPlan(
			std::uint64_t batch_sequence, bool retain_failed_candidates);
	void processRelocationBatch(bool retain_failed_candidates);
	void requeueCandidates(std::vector<PromotionCandidate> candidates);

	// The actual (was Controller::evictAnchor()) reclaim of every Region
	// dependency `anchor_id` holds -- unlike releaseAnchor() above, this runs
	// entirely synchronously including the GPU write-back/free, since it's
	// only ever called from the Coordinator's own thread
	// (processPromotions()'s capacity-retry loop), which needs the freed
	// capacity immediately. Also erases `anchor_id` from RoutingCache.
	void evictAnchorNow(VectorId anchor_id);
	std::vector<Region> retireAnchorsNow(const std::vector<VectorId>& anchor_ids);

	// Shared by evictAnchorNow() and the Coordinator's periodic draining of
	// releaseAnchor()'s deferred reclaims: batched write-back (if dirty) then
	// free() for every GPU-resident Region in `snapshots`.
	void reclaimRegions(const std::vector<Region>& snapshots);
	ReusableAllocations reclaimRegionsForPlan(
			const std::vector<Region>& snapshots,
			const std::vector<PromotionStorageRequest>& requests);
	void writeBackDirtyRegions(const std::vector<Region>& regions);

	// The Coordinator thread body: wakes every trigger_interval (or
	// immediately if force-woken by waitIdle()/shutdown()). Every wakeup
	// drains pending_promotions_/pending_reclaims_ regardless of what happens
	// next; processPromotions() itself only runs when force-woken or
	// replacement_policy_->onRelocationTrigger() says yes. See the
	// request-flow diagram in the file-level overview above.
	void coordinatorLoop();

	mutable RegionManagerMutex mutex_;
	std::unordered_map<RegionId, Region> regions_;
	std::unordered_map<RegionId, std::unordered_set<VectorId>> dependents_;
	std::unordered_map<VectorId, std::unordered_set<RegionId>> dependencies_;

	// Anchor-group tracking for group-based eviction (see
	// CoordinatorConfig::group_merge_overlap_threshold/max_eviction_group_size
	// and assignAnchorToGroup()'s own doc comment). Purely a clustering
	// *hint* layered on top of dependents_/dependencies_ above -- those two
	// remain the sole ground truth buildEvictionCandidates() actually trusts
	// for what's reclaimable, so staleness or a suboptimal grouping choice
	// here can only make eviction less effective, never unsafe. All three
	// guarded by mutex_ (same as dependents_/dependencies_): assignAnchorToGroup()
	// runs only on the Coordinator thread, but forget() -- which removes an
	// Anchor from its group -- is also reachable from releaseAnchor(), which a
	// Controller-calling (worker) thread can invoke via commitRemove().
	using GroupId = std::uint64_t;
	std::unordered_map<RegionId, GroupId> region_group_;             // last group a Region's promotion joined
	std::unordered_map<GroupId, std::unordered_set<VectorId>> group_members_;
	std::unordered_map<VectorId, GroupId> anchor_group_;
	GroupId next_group_id_ = 1;

	// Assigns `anchor_id` (whose current full dependency footprint is
	// `footprint_regions`) to a group: joins whichever existing group its
	// footprint overlaps most with, if that overlap covers at least
	// coordinator_config_.group_merge_overlap_threshold of the Anchor's own
	// footprint and the group has room (coordinator_config_.
	// max_eviction_group_size), else starts a new singleton group. Called
	// once per Anchor, immediately after its RoutingCache registration (see
	// processRelocationBatch()) -- an Anchor already in a group (e.g. a
	// re-promotion after residency was re-validated) is left in place, with
	// its footprint's Regions simply re-tagged to that same group.
	void assignAnchorToGroup(VectorId anchor_id, const std::vector<RegionId>& footprint_regions);

	// Per-Anchor epoch (see the epoch staleness invariant above), bumped only
	// by releaseAnchor(). Entries are never removed -- Controller::
	// MintAnchorId() never reuses a released Anchor's id on purpose anymore
	// (every mint is fresh -- see its own doc comment), but the *counter* it
	// draws from can in principle wrap all the way around and reissue an old
	// payload once every possible id has been minted (isKnownAnchor() is the
	// guard against exactly that), so a released id's epoch must stay
	// remembered forever for that guard to see it -- unbounded but
	// slow-growing, acceptable at this codebase's scale.
	std::unordered_map<VectorId, std::uint64_t> anchor_epoch_;

	// Strategy (design point 4): see ReplacementPolicy's doc comment. Never
	// touched directly by requestPromotion()/any Controller-calling thread --
	// only the Coordinator calls into it, except onAnchorEvicted() (from
	// releaseAnchor()) and onAnchorTouched() (from recordTraversal()).
	std::unique_ptr<ReplacementPolicy> replacement_policy_;

	// Coordinator wiring -- owned outside RegionManager, valid only between
	// start() and shutdown().
	IAdapter* adapter_ = nullptr;
	gpu::DeviceRegionPool* device_region_pool_ = nullptr;
	RoutingCache* routing_cache_ = nullptr;
	CoordinatorConfig coordinator_config_;

	std::thread coordinator_;
	RegionManagerCondVar coordinator_cv_;  // the Coordinator's own wake signal
	RegionManagerCondVar idle_cv_;         // what waitIdle() blocks on
	bool coordinator_running_ = false;
	bool coordinator_stop_requested_ = false;
	bool coordinator_force_wake_ = false;  // set by waitIdle()/shutdown() to skip the trigger_interval wait
	bool coordinator_reclaim_ready_ = false;  // a retiring Region reached zero logical pins
	bool coordinator_busy_ = false;        // true while a batch is actively being processed (queues already drained)
	std::deque<PromotionCandidate> pending_promotions_;  // RegionManager's own MPSC intake queue
	std::deque<Region> pending_reclaims_;
	std::atomic<std::uint64_t> next_candidate_sequence_{1};
	std::uint64_t next_batch_sequence_ = 1;  // Coordinator-thread-only

	// Stats' backing counters -- see stats() above. Independent atomics
	// rather than a mutex-guarded struct: each is bumped from a different
	// call site and read independently by stats(), with no cross-field
	// invariant requiring them to update as one atomic unit.
	std::atomic<std::uint64_t> stat_regions_promoted_{0};
	std::atomic<std::uint64_t> stat_regions_evicted_{0};
	std::atomic<std::uint64_t> stat_regions_written_back_{0};
	std::atomic<std::uint64_t> stat_anchor_evictions_{0};
	std::atomic<std::uint64_t> stat_compactions_total_{0};
	std::atomic<std::uint64_t> stat_relocation_batches_{0};
	std::atomic<std::uint64_t> stat_candidates_requeued_{0};
	std::atomic<std::uint64_t> stat_near_fit_reuses_{0};
	std::atomic<std::uint64_t> stat_candidates_rejected_{0};
};

}  // namespace arachne
