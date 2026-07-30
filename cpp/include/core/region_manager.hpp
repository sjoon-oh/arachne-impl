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
#include "types.hpp"

namespace arachne {

/// The single physical-mapping record for one RegionId: where its data
/// lives in host memory (`host`, reported by the adapter via
/// IRegion::hostView() -- see adapter/region.hpp), where it lives on GPU
/// once promoted (`device`, valid only after RegionManager actually
/// allocates through gpu::DeviceRegionPool), and the GPU write authority
/// currently held over it (`lease`, valid only while promoted). There is
/// exactly one Region per RegionId inside RegionManager -- unlike the old
/// per-Anchor Stitch, multiple Anchors that depend on the same Region all
/// see the same host/device/lease state, because there is only one copy of
/// it.
struct Region {
	RegionId id = 0;
	HostRegionView host;
	gpu::DeviceRegionHandle device;
	LeaseHandle lease;
};

/// Configuration for RegionManager's Coordinator thread (see start()).
/// `trigger_interval` bounds how often the Coordinator wakes up to drain its
/// intake queues and offer the replacement policy a chance to run a
/// relocation batch -- it is *not* itself the decision of whether to
/// actually run one; that is ReplacementPolicy::onRelocationTrigger()'s job
/// (see its doc comment). requestPromotion()/releaseAnchor() deliberately do
/// *not* wake the Coordinator early on their own (see coordinatorLoop()'s
/// doc comment) -- only a periodic tick or an explicit waitIdle()/
/// shutdown() call does.
struct CoordinatorConfig {
	std::chrono::milliseconds trigger_interval{100};
};

/// Owns every registered Region, the bipartite Anchor<->Region dependency
/// graph, the Anchor replacement policy, and (once start() is called) the
/// Coordinator: a single dedicated background thread that actually performs
/// GPU residency management -- Promotion, Eviction, Write-back, Compaction
/// -- lazily, batched, off of any Controller-calling thread's own critical
/// path. This used to be split across Controller (which owned
/// replacement_policy_ and did all of make()/promoteAnchor()/evictAnchor()/
/// writeBackDirtyRegions() synchronously, inline in insert()/remove()) and
/// RegionManager (pure bookkeeping); it is now unified here, since "which
/// Anchor should be evicted, which Region promoted next, and when" is a
/// single cohesive policy question RegionManager is best placed to own --
/// Controller is left responsible only for routing (design point 1) and
/// scheduling (OpScheduler), not GPU residency policy.
///
/// Two things stay strictly synchronous (never deferred to the
/// Coordinator), because callers depend on seeing their effect immediately:
///  - The bipartite dependency graph itself (registerRegion()/
///    addDependency()/forget()/etc, all unchanged from before) -- so
///    regionsOf()/route()'s GPU-only-routing decision is never stale.
///  - releaseAnchor()'s write-lease release and clearResidency(), and its
///    replacement_policy_->onAnchorEvicted() notification -- so a Region
///    orphaned by one Anchor is immediately eligible to be re-promoted fresh
///    for a different Anchor, without waiting for the Coordinator to
///    actually free its old GPU allocation first (see releaseAnchor()'s doc
///    comment for why this matters).
///
/// Everything else -- requestPromotion()'s enqueue, and every subsequent
/// step (handing candidates to the replacement policy, deciding whether and
/// what to promote/evict, the actual GPU work) -- happens only on the
/// Coordinator thread. requestPromotion() is a pure multiple-producer
/// append onto RegionManager's own intake queue; the Coordinator is the
/// queue's *only* consumer, draining it and handing each candidate to
/// replacement_policy_->enqueueCandidate() -- from that point the candidate
/// lives only inside the policy's own storage. This keeps every
/// replacement-policy decision (what to promote, what to evict, whether now
/// is even the right time -- see ReplacementPolicy::onRelocationTrigger())
/// on one single thread, so a policy implementation never has to reason
/// about concurrent callers touching its internal ranking state -- only
/// onAnchorEvicted() and onAnchorTouched() are still called from whatever
/// thread the underlying event happens on (see ReplacementPolicy's own doc
/// comment for why those two are the exceptions).
///
/// RoutingCache ownership: RegionManager now owns *writing* to RoutingCache
/// (Controller keeps reading it, via RoutingCache::nearest() in route()) --
/// ensure() is called from processPromotions(), only once a candidate's
/// footprint actually gained at least one Region dependency, and erase() is
/// called from both releaseAnchor() and evictAnchorNow(), whenever an Anchor
/// stops being GPU-resident for any reason (delete, verification mismatch,
/// or plain capacity-driven eviction). Previously this lived in Controller's
/// commitSearch()/commitInsert(), unconditionally, independent of whether
/// promotion actually happened -- see PromotionCandidate's own doc comment
/// (replacement_policy.hpp) for the vector-lifetime and epoch-staleness
/// machinery this move required.
///
/// Thread-safety: every bookkeeping method takes RegionManager's own lock,
/// mirroring the pre-Coordinator contract (Controller is called
/// concurrently). requestPromotion()/releaseAnchor()/waitIdle() are safe to
/// call from any number of concurrent Controller-calling threads; the
/// Coordinator itself is the only thread that ever performs actual GPU
/// work, so that work is inherently serialized without needing its own
/// separate lock.
class RegionManager {
 public:
	/// `replacement_policy` defaults to FifoReplacementPolicy when left null,
	/// mirroring OpScheduler's own SchedulingPolicy default -- see
	/// ReplacementPolicy's doc comment.
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

	/// Records `anchor_id` as a dependent of `region_id` (idempotent -- a
	/// second call for the same pair changes nothing). Returns false without
	/// recording anything if `region_id` was never registered -- callers
	/// (make()) are expected to check this before attempting to promote a
	/// region nobody opted in.
	bool addDependency(VectorId anchor_id, RegionId region_id);

	/// Drops `anchor_id`'s dependency on `region_id`. Returns true exactly
	/// when that was the *last* Anchor depending on it -- the caller's signal
	/// to actually reclaim the lease/device residency via clearResidency()
	/// below, since other Anchors may still be relying on this Region staying
	/// promoted. Returns false (and changes nothing) if there was no such
	/// dependency, or if other Anchors still depend on `region_id`.
	bool removeDependency(VectorId anchor_id, RegionId region_id);

	/// Drops every dependency `anchor_id` has, returning the RegionIds that
	/// consequently dropped to zero dependents -- the set the caller should
	/// reclaim (release lease, free device memory, clearResidency()). Regions
	/// still depended on by some other Anchor are not included, matching
	/// removeDependency()'s per-pair semantics.
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
	/// owned outside RegionManager (by Controller) and must outlive it,
	/// exactly like OpScheduler's `adapter_` pointer. `routing_cache` is
	/// where the Coordinator registers/erases Anchors as their GPU residency
	/// actually changes -- see the class doc comment above. Must be called
	/// before requestPromotion()/releaseAnchor() are used. Throws
	/// std::logic_error if already started.
	void start(IAdapter& adapter, gpu::DeviceRegionPool& device_region_pool, RoutingCache& routing_cache,
						 CoordinatorConfig config = {});

	/// Stops the Coordinator, draining whatever it's currently processing (or
	/// picks up in one last pass) before returning -- mirrors
	/// OpScheduler::shutdown(). Safe to call if never started, or already
	/// shut down (no-op).
	void shutdown();

	/// Replaces the old, eager Controller::promoteAnchor(): appends a
	/// PromotionCandidate (built from `anchor_id`/`footprint`/`vector`) onto
	/// RegionManager's own intake queue and returns immediately -- a pure
	/// enqueue, safe to call concurrently from any number of caller threads
	/// (multiple-producer). Does *not* touch replacement_policy_ itself: only
	/// the Coordinator thread (the queue's single consumer) ever hands
	/// candidates to the policy, via ReplacementPolicy::enqueueCandidate() --
	/// see the class doc comment above for why.
	///
	/// `vector` is copied into the enqueued candidate immediately (see
	/// PromotionCandidate's own doc comment for why an owned copy is
	/// required); `vector.data` need only stay valid for the duration of
	/// this call. Defaults to an empty VectorView for callers (mainly
	/// RegionManager's own lower-level tests) that never exercise the
	/// RoutingCache-registration path this candidate eventually drives.
	///
	/// The candidate is stamped with `anchor_id`'s current epoch (see
	/// releaseAnchor()) -- if `anchor_id` is released before the Coordinator
	/// gets around to granting this candidate, the epoch mismatch causes it
	/// to be discarded rather than promoted/registered.
	void requestPromotion(VectorId anchor_id, RegionFootprint footprint, VectorView vector = {});

	/// Replaces the old, eager Controller::evictAnchor() call sites (delete,
	/// verification mismatch). Immediately drops every Region dependency
	/// `anchor_id` has (forget()), releases the write lease on each Region
	/// that consequently has zero remaining dependents, clears its residency
	/// back to host-only, and erases `anchor_id` from RoutingCache (see the
	/// class doc comment above) -- all synchronous, so a subsequent
	/// requestPromotion() for a *different* Anchor wanting the same Region is
	/// never blocked behind, or confused by, this Region's still-pending GPU
	/// reclaim. Only the actual write-back-if-dirty + free of the now-orphaned
	/// device memory is deferred to the Coordinator (batched with whatever
	/// else is pending at its next trigger).
	///
	/// Also bumps `anchor_id`'s epoch (see PromotionCandidate's own doc
	/// comment): any PromotionCandidate for `anchor_id` already enqueued but
	/// not yet granted becomes stale and will be discarded by
	/// processPromotions() rather than acted on -- this is what keeps a
	/// deleted-then-VectorId-reused Anchor from having a stale promotion
	/// request revive its old dependency-graph/RoutingCache state under the
	/// new insert's identity.
	void releaseAnchor(VectorId anchor_id);

	/// Reports that a traversal (Controller::dispatch(TraverseRequest), the
	/// common entry point behind both search() and insert()'s own placement
	/// lookup) actually accessed every Region in `touched` -- forwarded to
	/// the replacement policy as an onAnchorTouched() call for every Anchor
	/// currently depending on one of those Regions, deduplicated so an Anchor
	/// depending on several touched Regions is only reported once per call.
	/// Regions with no registered dependents (never registered, or currently
	/// depended on by nobody) are silently skipped -- this is a best-effort
	/// hotness signal, not bookkeeping, so it never throws. Synchronous and
	/// cheap (bounded by `touched.regions.size()`, a mutex-guarded map
	/// lookup per entry): unlike requestPromotion()/releaseAnchor(), there is
	/// no GPU work here for the Coordinator to defer.
	void recordTraversal(const RegionFootprint& touched);

	/// Blocks until the Coordinator has fully drained everything requested
	/// via requestPromotion()/releaseAnchor() so far (including anything
	/// requested concurrently while this call is waiting) -- for callers
	/// (tests, or an operator wanting a synchronous checkpoint) that need a
	/// guarantee GPU residency actually reflects every request made, not just
	/// that the request was accepted. Forces the Coordinator to wake
	/// immediately rather than waiting out its own trigger_interval, so this
	/// is fast even with a long interval configured. Not needed on the
	/// normal, fully-async path.
	void waitIdle();

	/// Snapshot of RegionManager's own internal counters -- the counterpart
	/// to the ARACHNE_LOG_DEBUG lines threaded through make()/
	/// evictAnchorNow()/writeBackDirtyRegions(), for callers (tests,
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
	// Outcome of make() below, distinguishing two very different reasons a
	// Region can fail to become available for `anchor_id`: processPromotions()
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
	// below in place of a bare gpu::DeviceRegionPool::tryAllocate() call --
	// see the original Controller::allocateWithCompaction()'s doc comment for
	// the tryAllocate-then-compact-then-retry shape (unchanged). `pending` is
	// flushed and cleared before compact() runs if non-empty -- see the .cpp
	// for why: compact() touches every live Data-kind allocation, including
	// ones already promoted earlier in the same batch and still holding an
	// open Lease in `pending`, which would otherwise deadlock compact()'s own
	// wait against a Lease this same call chain is still holding open.
	std::optional<gpu::DeviceRegionHandle> allocateWithCompaction(std::size_t bytes,
																																 std::vector<gpu::DeviceRegionPool::Lease>& pending);

	// Region promotion (design point 4): acquires a write lease and enqueues
	// (does not wait for) the host-to-device copy of `region`'s data if it
	// isn't already promoted by some other Anchor, then records `anchor_id`
	// as a dependent. See MakeResult above for what a non-Promoted return
	// means. Does *not* notify replacement_policy_ -- selectNextPromotionCandidate()
	// already recorded `anchor_id` for eviction-ordering purposes, before this
	// call ever runs (see ReplacementPolicy::selectNextPromotionCandidate()'s
	// own doc comment). Does *not* touch RoutingCache either -- a candidate's
	// footprint can span several Regions, so processPromotions() is the one
	// that decides, once every Region in a candidate's footprint has been
	// attempted, whether the Anchor gained enough residency to be worth
	// registering (see its own doc comment).
	MakeResult make(VectorId anchor_id, RegionId region, std::vector<gpu::DeviceRegionPool::Lease>& pending,
								 std::vector<std::vector<std::byte>>& zero_headers);

	// `anchor_id`'s current epoch, 0 if it has never had one (releaseAnchor()
	// is the only thing that bumps it) -- see PromotionCandidate's own doc
	// comment. Caller must already hold mutex_.
	std::uint64_t currentEpochLocked(VectorId anchor_id) const;

	// Drives one Coordinator pass's worth of promotion work by repeatedly
	// pulling from replacement_policy_->selectNextPromotionCandidate() until
	// it returns nullopt -- see that method's doc comment for why there is no
	// separate "batch" argument here anymore: the policy itself decides how
	// many candidates (out of whatever it has admitted via enqueueCandidate())
	// to offer this round. Each candidate is first checked against
	// currentEpochLocked() -- a mismatch means `anchor_id` was released
	// (deleted, or verification-reset) since this candidate was enqueued, so
	// it's discarded outright (no make() call, no RoutingCache registration).
	// Otherwise retries each OutOfCapacity result via
	// selectNextEvictionCandidate()-driven eviction, then -- once every Region
	// in the candidate's footprint has been attempted -- registers the Anchor
	// in RoutingCache (routing_cache_->ensure()) iff at least one Region
	// actually became a dependency, using the candidate's own owned vector
	// copy (see PromotionCandidate::vectorView()). Finally flushes every
	// host-to-device copy enqueued across the *entire* pass in one
	// gpu::DeviceRegionPool::flush() call, not one per Anchor.
	void processPromotions();

	// The actual (was Controller::evictAnchor()) reclaim of every Region
	// dependency `anchor_id` currently holds -- unlike releaseAnchor() above,
	// this runs entirely synchronously, including the GPU write-back/free,
	// since it's only ever called from within the Coordinator's own thread
	// (processPromotions()'s capacity-retry loop), which needs the freed
	// capacity immediately to retry make(). Also erases `anchor_id` from
	// RoutingCache -- see the class doc comment above.
	void evictAnchorNow(VectorId anchor_id);

	// Shared by evictAnchorNow() and the Coordinator's periodic draining of
	// releaseAnchor()'s deferred reclaims: batched write-back (if dirty) then
	// free() for every GPU-resident Region in `snapshots` -- was
	// Controller::writeBackDirtyRegions() plus the free()-loop tail of
	// evictAnchor(), combined and relocated unchanged.
	void reclaimRegions(const std::vector<Region>& snapshots);
	void writeBackDirtyRegions(const std::vector<Region>& regions);

	// The Coordinator thread body: wakes every CoordinatorConfig::
	// trigger_interval (or immediately if force-woken by waitIdle()/
	// shutdown()). Every wakeup, regardless of what happens next: drains
	// pending_promotions_ (RegionManager's own MPSC intake queue) and hands
	// each candidate to replacement_policy_->enqueueCandidate() -- this is
	// the "background work" that keeps happening continuously rather than
	// only starting once a trigger fires -- and drains+processes
	// pending_reclaims_ exactly as before (unrelated to the replacement
	// policy; already-decided cleanup work, not a policy decision). Then, only
	// if force-woken (waitIdle()/shutdown(), which must guarantee everything
	// requested so far has actually happened) or
	// replacement_policy_->onRelocationTrigger() says yes, runs
	// processPromotions(). Deliberately never woken early by
	// requestPromotion()/releaseAnchor() themselves -- only a periodic tick or
	// an explicit waitIdle()/shutdown() call does.
	void coordinatorLoop();

	mutable std::mutex mutex_;
	std::unordered_map<RegionId, Region> regions_;
	std::unordered_map<RegionId, std::unordered_set<VectorId>> dependents_;
	std::unordered_map<VectorId, std::unordered_set<RegionId>> dependencies_;

	// Per-Anchor epoch (see PromotionCandidate's own doc comment) -- bumped
	// only by releaseAnchor(). Entries are never removed (an Anchor id can be
	// released and later reused by a new insert(), and the epoch must keep
	// increasing across that reuse) -- unbounded but slow-growing relative to
	// the id space itself, acceptable at this codebase's scale.
	std::unordered_map<VectorId, std::uint64_t> anchor_epoch_;

	// Strategy (design point 4): see ReplacementPolicy's doc comment. Never
	// touched directly by requestPromotion()/any Controller-calling thread --
	// only the Coordinator thread calls into it, except for onAnchorEvicted()
	// (from releaseAnchor(), immediate) and onAnchorTouched() (from
	// recordTraversal(), immediate) -- see ReplacementPolicy's own doc comment.
	std::unique_ptr<ReplacementPolicy> replacement_policy_;

	// Coordinator wiring -- owned outside RegionManager, valid only between
	// start() and shutdown().
	IAdapter* adapter_ = nullptr;
	gpu::DeviceRegionPool* device_region_pool_ = nullptr;
	RoutingCache* routing_cache_ = nullptr;
	CoordinatorConfig coordinator_config_;

	std::thread coordinator_;
	std::condition_variable coordinator_cv_;  // the Coordinator's own wake signal
	std::condition_variable idle_cv_;         // what waitIdle() blocks on
	bool coordinator_running_ = false;
	bool coordinator_stop_requested_ = false;
	bool coordinator_force_wake_ = false;  // set by waitIdle()/shutdown() to skip the trigger_interval wait
	bool coordinator_busy_ = false;        // true while a batch is actively being processed (queues already drained)
	std::deque<PromotionCandidate> pending_promotions_;  // RegionManager's own MPSC intake queue
	std::deque<Region> pending_reclaims_;

	// Stats' backing counters -- see stats() and Stats' own doc comment.
	// Independent atomics rather than a mutex-guarded struct: each is bumped
	// from a different call site (make(), reclaimRegions(), releaseAnchor())
	// and read independently by stats(), so there's no cross-field invariant
	// that would need them updated as one atomic unit.
	std::atomic<std::uint64_t> stat_regions_promoted_{0};
	std::atomic<std::uint64_t> stat_regions_evicted_{0};
	std::atomic<std::uint64_t> stat_regions_written_back_{0};
	std::atomic<std::uint64_t> stat_anchor_evictions_{0};
	std::atomic<std::uint64_t> stat_compactions_total_{0};
};

}  // namespace arachne
