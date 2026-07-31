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
struct Region {
	RegionId id = 0;
	HostRegionView host;
	gpu::DeviceRegionHandle device;
	LeaseHandle lease;
};

/// Configuration for the Coordinator thread (see start()). `trigger_interval`
/// only bounds how often the Coordinator wakes to drain its intake queues and
/// offer the replacement policy a chance to run; the decision of whether to
/// actually run a relocation batch is ReplacementPolicy::onRelocationTrigger()'s
/// alone. Queue-append calls never wake the Coordinator early on their own.
struct CoordinatorConfig {
	std::chrono::milliseconds trigger_interval{100};
};

/// GPU residency policy owner -- see the file-level overview above for
/// RegionManager's design, invariants, and the Coordinator request-flow
/// diagram.
class RegionManager {
 public:
	/// `replacement_policy` defaults to FifoReplacementPolicy when left null,
	/// mirroring OpScheduler's own SchedulingPolicy default.
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
	};

	// Capacity-aware allocation with a compaction fallback (tryAllocate ->
	// compact -> retry), used by make() in place of a bare
	// gpu::DeviceRegionPool::tryAllocate() call. See the "allocateWithCompaction()
	// fallback chain" invariant in the file-level overview above for why
	// `pending` must be flushed before compact() runs.
	std::optional<gpu::DeviceRegionHandle> allocateWithCompaction(std::size_t bytes,
																																 std::vector<gpu::DeviceRegionPool::Lease>& pending);

	// Region promotion (design point 4): acquires a write lease and enqueues
	// (does not wait for) the host-to-device copy of `region`'s data if it
	// isn't already promoted, then records `anchor_id` as a dependent. See
	// MakeResult above for what a non-Promoted return means. Does not touch
	// replacement_policy_ or RoutingCache -- selectNextPromotionCandidate()
	// and processPromotions() own those, respectively (see their own doc
	// comments).
	MakeResult make(VectorId anchor_id, RegionId region, std::vector<gpu::DeviceRegionPool::Lease>& pending,
								 std::vector<std::vector<std::byte>>& zero_headers);

	// `anchor_id`'s current epoch, 0 if it has never had one (releaseAnchor()
	// is the only thing that bumps it). Caller must already hold mutex_.
	std::uint64_t currentEpochLocked(VectorId anchor_id) const;

	// Drives one Coordinator pass's worth of promotion work: repeatedly pulls
	// from replacement_policy_->selectNextPromotionCandidate() until nullopt,
	// discards epoch-stale candidates, retries OutOfCapacity results via
	// eviction (see the OutOfCapacity retry loop invariant above), then
	// registers each Anchor in RoutingCache iff it gained at least one Region
	// dependency. Flushes every host-to-device copy from the whole pass once.
	void processPromotions();

	// The actual (was Controller::evictAnchor()) reclaim of every Region
	// dependency `anchor_id` holds -- unlike releaseAnchor() above, this runs
	// entirely synchronously including the GPU write-back/free, since it's
	// only ever called from the Coordinator's own thread
	// (processPromotions()'s capacity-retry loop), which needs the freed
	// capacity immediately. Also erases `anchor_id` from RoutingCache.
	void evictAnchorNow(VectorId anchor_id);

	// Shared by evictAnchorNow() and the Coordinator's periodic draining of
	// releaseAnchor()'s deferred reclaims: batched write-back (if dirty) then
	// free() for every GPU-resident Region in `snapshots`.
	void reclaimRegions(const std::vector<Region>& snapshots);
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

	// Per-Anchor epoch (see the epoch staleness invariant above), bumped only
	// by releaseAnchor(). Entries are never removed -- an Anchor id can be
	// released and later reused by a new insert(), and the epoch must keep
	// increasing across that reuse -- unbounded but slow-growing, acceptable
	// at this codebase's scale.
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
	bool coordinator_busy_ = false;        // true while a batch is actively being processed (queues already drained)
	std::deque<PromotionCandidate> pending_promotions_;  // RegionManager's own MPSC intake queue
	std::deque<Region> pending_reclaims_;

	// Stats' backing counters -- see stats() above. Independent atomics
	// rather than a mutex-guarded struct: each is bumped from a different
	// call site and read independently by stats(), with no cross-field
	// invariant requiring them to update as one atomic unit.
	std::atomic<std::uint64_t> stat_regions_promoted_{0};
	std::atomic<std::uint64_t> stat_regions_evicted_{0};
	std::atomic<std::uint64_t> stat_regions_written_back_{0};
	std::atomic<std::uint64_t> stat_anchor_evictions_{0};
	std::atomic<std::uint64_t> stat_compactions_total_{0};
};

}  // namespace arachne
