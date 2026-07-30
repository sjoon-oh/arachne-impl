#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

#include "adapter/region.hpp"
#include "types.hpp"

namespace arachne {

/// One <anchor, regions> pair RegionManager::requestPromotion() enqueues.
/// Shared between RegionManager and ReplacementPolicy so a policy can hold
/// and manage its own copy of admitted candidates (see
/// ReplacementPolicy::enqueueCandidate()) without RegionManager needing to
/// keep a second, redundant copy of the same data once it has been handed
/// off.
///
/// `vector_bytes`/`vector_dim`/`vector_dtype` are an *owned* copy of the
/// Anchor's own vector -- RegionManager now registers the Anchor in
/// RoutingCache itself, at actual promotion-grant time (see
/// RegionManager::processPromotions()), which can happen an arbitrary
/// amount of time (bounded by CoordinatorConfig::trigger_interval) after
/// requestPromotion() was called. VectorView is explicitly non-owning and
/// not guaranteed to outlive a single call (see its own doc comment in
/// types.hpp), so the bytes must be copied out into this candidate at
/// enqueue time, while the original caller-owned buffer is still guaranteed
/// alive -- see Controller::dispatch()'s own doc comment for why that's
/// still true even though this runs on an OpScheduler worker thread rather
/// than the original calling thread.
///
/// `epoch` guards against a second, independent race the deferred-ensure()
/// design above introduces: `anchor_id` could be deleted (Controller::remove())
/// -- or, for an insert-derived anchor_id, deleted *and have its VectorId
/// reused by a brand-new, unrelated insert() -- in the window between this
/// candidate being enqueued and the Coordinator actually granting it.
/// RegionManager stamps the Anchor's current epoch (bumped by
/// releaseAnchor(), see its own doc comment) onto the candidate at
/// requestPromotion() time; if that epoch no longer matches at grant time,
/// the candidate is stale and is discarded rather than promoted/registered.
struct PromotionCandidate {
	VectorId anchor_id = 0;
	RegionFootprint footprint;
	std::uint64_t epoch = 0;
	std::vector<std::byte> vector_bytes;
	std::uint32_t vector_dim = 0;
	VectorDType vector_dtype = VectorDType::Float32;

	/// Reconstructs a VectorView over this candidate's own owned copy --
	/// valid for as long as this PromotionCandidate itself is (unlike a
	/// VectorView built from the original caller's buffer, which is not).
	VectorView vectorView() const { return VectorView{vector_bytes.data(), vector_dim, vector_dtype}; }
};

/// Pluggable Anchor-level Region replacement policy (Quick Summary design
/// point 4): decides which Anchor's Region dependencies to promote next and
/// which to reclaim to make room. Always operates on Anchor ids, never on
/// individual Regions -- an Anchor and every Region RegionManager currently
/// says it depends on (see core/region_manager.hpp) are the unit of
/// locality this policy reasons about, per the Anchor-centric residency
/// design (replacement is about which *Anchor* has gone cold, not which
/// Region looks sparse).
///
/// Mirrors SchedulingPolicy's shape (core/scheduling_policy.hpp): a pure
/// interface here, concrete strategies (FifoReplacementPolicy today; LRU or
/// a GPU-aware hotness/latency/transfer-cost score later) as separate
/// classes, RegionManager only ever calling through this interface -- and
/// RegionManager owns the concrete instance the same way OpScheduler owns a
/// SchedulingPolicy (std::unique_ptr, defaulted to Fifo* when none is
/// injected).
///
/// Threading model (this is the part that changed from the original,
/// simpler design -- see the Coordinator doc comment in
/// core/region_manager.hpp for the full rationale): RegionManager's own
/// intake queue (fed by requestPromotion(), called from any number of
/// concurrent caller threads -- search/insert workers) is a
/// multiple-producer/single-consumer queue whose *only* consumer is the
/// Coordinator thread. The Coordinator drains it and hands each item to
/// this policy via enqueueCandidate() -- from that point on, the candidate
/// exists nowhere but inside the policy's own storage, and only the
/// Coordinator thread ever calls into this policy again (via
/// onRelocationTrigger()/selectNextPromotionCandidate()/
/// selectNextEvictionCandidate()) to work through it. The two exceptions,
/// still called from whatever thread the underlying event happens on
/// (never funneled through the Coordinator), are onAnchorEvicted() (an
/// Anchor's dependencies were just dropped -- callers depend on this being
/// reflected immediately, the same way RegionManager's own dependency graph
/// bookkeeping is immediate) and onAnchorTouched() (a pure hotness signal
/// with no ordering/membership contract to protect). A policy implementation
/// must make every method thread-safe against this mixed calling pattern.
class ReplacementPolicy {
 public:
	virtual ~ReplacementPolicy() = default;

	/// The consumer side of RegionManager's own MPSC intake queue: transfers
	/// ownership of one promotion candidate into the policy's own storage.
	/// Called only by the Coordinator thread, once per candidate, as it
	/// drains RegionManager's pending_promotions_ -- see the class doc
	/// comment above. After this call, RegionManager keeps no copy of
	/// `candidate`; the policy is the sole owner until it returns it (or
	/// chooses to discard it) via selectNextPromotionCandidate().
	virtual void enqueueCandidate(PromotionCandidate candidate) = 0;

	/// Notifies the policy that `anchor_id` no longer depends on any Region --
	/// called immediately (never deferred to a trigger), whenever
	/// RegionManager learns this, whether via an explicit releaseAnchor()
	/// (delete, verification mismatch) or evictAnchorNow() (capacity-driven,
	/// Coordinator-thread-only). Must purge `anchor_id` from *every*
	/// structure the policy maintains: both admitted-but-not-yet-selected
	/// candidates (in case a promotion for `anchor_id` is still sitting
	/// unselected -- e.g. requested, then deleted, before the Coordinator got
	/// to it) and whatever eviction-ordering structure
	/// selectNextPromotionCandidate() populated (in case it was already
	/// granted). No-op if the policy isn't tracking `anchor_id` at all.
	virtual void onAnchorEvicted(VectorId anchor_id) = 0;

	/// Notifies the policy that `anchor_id` was a dependent of a Region a
	/// traversal (search's own lookup, or insert's placement lookup -- see
	/// core/region_manager.hpp's RegionManager::recordTraversal(), which
	/// derives this from the RegionFootprint every dispatch() call returns)
	/// actually accessed. This is a pure hotness/recency signal, distinct
	/// from candidate/eviction-order membership -- a policy that only orders
	/// by promotion order (FifoReplacementPolicy) is free to ignore it
	/// entirely. Called synchronously from whichever thread performed the
	/// traversal, *not* funneled through the Coordinator's single-consumer
	/// queue, since there is no ordering/membership contract to protect
	/// here. May fire multiple times for the same `anchor_id` within what a
	/// caller considers one logical traversal if that traversal touched more
	/// than one Region depended on by the same Anchor; RegionManager
	/// deduplicates per call to recordTraversal() but not across calls, so a
	/// policy that cares about firing frequency (not just recency) should
	/// account for that itself.
	virtual void onAnchorTouched(VectorId anchor_id) = 0;

	/// Called by the Coordinator once per wakeup, before touching any
	/// pending/resident state, to decide whether this tick should actually
	/// perform a relocation batch (promotion + any eviction it needs) at
	/// all. Returning false costs nothing beyond this call -- the
	/// Coordinator goes back to sleep untouched, GPU-work-free, until its
	/// next wakeup. Ignored (bypassed) when the Coordinator has been
	/// force-woken by waitIdle()/shutdown() -- those callers need a
	/// guarantee that everything requested so far has actually happened,
	/// which requires overriding whatever timing preference the policy would
	/// otherwise apply.
	virtual bool onRelocationTrigger() = 0;

	/// True if the policy is currently holding any admitted-but-not-yet-
	/// promoted candidate (admitted via enqueueCandidate(), not yet returned
	/// by selectNextPromotionCandidate()). Used by RegionManager::waitIdle()
	/// to know whether draining its own intake queue to empty actually means
	/// "nothing left to do" -- a candidate can sit inside the policy's own
	/// storage, unselected, across one or more Coordinator ticks if
	/// onRelocationTrigger() keeps declining, or if the policy itself is
	/// deliberately holding some back (see selectNextPromotionCandidate()).
	virtual bool hasPendingCandidates() const = 0;

	/// Pops the next promotion candidate the policy has decided to act on
	/// this round, out of its own storage (populated via
	/// enqueueCandidate()). Called repeatedly by the Coordinator -- only
	/// after onRelocationTrigger() returned true (or the Coordinator was
	/// force-woken) -- until it returns nullopt, which ends this round's
	/// promotion pass however many candidates were actually returned before
	/// that; there is no separate "how many" query, the policy simply stops
	/// offering more once it decides to. Implementations are free to
	/// silently discard candidates they no longer consider meaningful
	/// without ever returning them (e.g. stale by the time they'd be
	/// considered) -- since the policy owns its only copy, nothing else
	/// needs to be told when that happens.
	///
	/// Expected (though not required) to record the returned candidate's
	/// anchor_id into whatever ordering structure
	/// selectNextEvictionCandidate() reads from, at the moment it decides to
	/// return it here -- there is no separate "promotion actually
	/// succeeded" confirmation call from RegionManager. A candidate returned
	/// here but later found NotEligible by RegionManager::make() (for every
	/// Region in its footprint) still counts as a tracked, selectable
	/// eviction candidate until onAnchorEvicted() removes it -- a bounded,
	/// self-correcting cost (evicting it reclaims nothing, but does
	/// permanently drop it from the pool, so a capacity-retry loop still
	/// makes monotonic progress and terminates), not a correctness bug.
	virtual std::optional<PromotionCandidate> selectNextPromotionCandidate() = 0;

	/// Chooses the next Anchor to reclaim to make room for a Promotion,
	/// excluding `excluded` (the Anchor currently being promoted -- a policy
	/// must never select the thing it's making room for). Returns nullopt if
	/// there is nothing eligible to evict.
	virtual std::optional<VectorId> selectNextEvictionCandidate(VectorId excluded) = 0;
};

/// Default policy: admits candidates in arrival order and reclaims whichever
/// currently-tracked Anchor was granted its first Region dependency longest
/// ago, irrespective of any Anchor-query hotness signal since. Deliberately
/// the simplest possible strategy -- it stands up the Eviction -> (optional)
/// Compaction -> Promotion pipeline's skeleton (see
/// gpu/device_region_pool.hpp's compact() doc comment for the GPU memory
/// half of that pipeline) before a real Anchor-query-aware scoring policy
/// replaces it. Never declines a trigger or holds candidates back: as long
/// as it has any admitted candidates, onRelocationTrigger() says yes and
/// selectNextPromotionCandidate() offers every one of them, in the order
/// they were admitted, before returning nullopt.
class FifoReplacementPolicy final : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override;
	void onAnchorEvicted(VectorId anchor_id) override;

	/// No-op: FIFO orders purely by admission order, deliberately blind to
	/// usage frequency -- see the class doc comment above.
	void onAnchorTouched(VectorId anchor_id) override;

	bool onRelocationTrigger() override;
	bool hasPendingCandidates() const override;
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override;
	std::optional<VectorId> selectNextEvictionCandidate(VectorId excluded) override;

 private:
	mutable std::mutex mutex_;
	std::deque<PromotionCandidate> pending_candidates_;  // admitted, not yet selected -- oldest-first
	std::deque<VectorId> promoted_order_;                // selected, eviction-selectable -- oldest-first
	std::unordered_set<VectorId> promoted_tracked_;      // dedup guard for promoted_order_
};

}  // namespace arachne
