#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapter/region.hpp"
#include "types.hpp"

namespace arachne {

// ReplacementPolicy is Arachne's pluggable strategy (Quick Summary design
// point 4) for deciding, per Anchor, which Region dependencies to promote
// into GPU residency next and which currently-resident Anchor to reclaim to
// make room. It never touches residency itself: RegionManager is the sole
// authority that grants a promotion or performs an eviction, and
// independently re-verifies eligibility (RegionManager::make()) before
// acting on anything a policy returns -- a policy only *suggests*. Every
// policy reasons at Anchor granularity, never per-Region, matching the
// Anchor-centric residency model: replacement is about which Anchor has
// gone cold, not which Region looks sparse. Concrete strategies (Fifo, LRU,
// LFU, Clock, 2Q below) share this one interface; RegionManager owns a
// single instance via std::unique_ptr the same way OpScheduler owns a
// SchedulingPolicy.
//
// Candidate lifecycle:
//
//   RegionManager::requestPromotion()        (any caller thread)
//            |
//            v
//   RegionManager's MPSC intake queue        (pending_promotions_)
//            |   drained only by the Coordinator thread
//            v
//   ReplacementPolicy::enqueueCandidate()    -- policy takes sole ownership
//            |
//            v
//   pending (policy-internal storage)
//            |   selectNextPromotionCandidate(), Coordinator thread only
//            v
//   granted / tracked  <----------------------------.
//            |                                       |
//            |-- selectNextEvictionCandidate() ------'  (reads tracked state)
//            |
//            |-- onAnchorEvicted()  (any thread, immediate) --> purged from
//            |                                                  every structure
//            '-- onAnchorTouched()  (any thread, immediate) --> hotness/recency
//                                                                updated in place
//
// PromotionCandidate is the one <anchor, regions> unit flowing through this
// pipeline. Its vector_bytes/vector_dim/vector_dtype are an *owned* copy of
// the Anchor's vector, not a VectorView -- RegionManager only registers the
// Anchor in RoutingCache at actual grant time (RegionManager::
// processPromotions()), an arbitrary delay after requestPromotion() returns,
// well past when the caller-owned buffer a VectorView would point to is
// still guaranteed alive (see Controller::dispatch()'s doc comment).
//
// `epoch` closes a second race that delay opens up: anchor_id can be deleted
// (Controller::remove()), or deleted and have its VectorId reused by an
// unrelated insert(), in the window between enqueueing and grant.
// RegionManager stamps the Anchor's current epoch (bumped by
// releaseAnchor()) onto the candidate at requestPromotion() time; a mismatch
// at grant time means the candidate is stale and must be discarded rather
// than promoted/registered.
//
// Threading model: enqueueCandidate() and every "act on pending/tracked
// state" call (onRelocationTrigger(), selectNextPromotionCandidate(),
// selectNextEvictionCandidate()) run only on the Coordinator thread, mirroring
// the intake queue's MPSC/single-consumer shape. onAnchorEvicted() and
// onAnchorTouched() are the exception: both run on whichever thread the
// underlying event happened on, never funneled through the Coordinator --
// eviction bookkeeping must be immediate (matching RegionManager's own
// dependency-graph updates), and touch has no ordering/membership contract to
// protect. Every concrete policy must be thread-safe against this mix.

/// One <anchor, regions> pair enqueued by RegionManager::requestPromotion();
/// see the file overview above for why vector_bytes is an owned copy and how
/// `epoch` guards against staleness.
struct PromotionCandidate {
	VectorId anchor_id = 0;
	RegionFootprint footprint;
	std::uint64_t epoch = 0;
	std::vector<std::byte> vector_bytes;
	std::uint32_t vector_dim = 0;
	VectorDType vector_dtype = VectorDType::Float32;

	/// View over this candidate's own owned copy -- valid for as long as this
	/// PromotionCandidate is, unlike a VectorView into the original caller's
	/// buffer.
	VectorView vectorView() const { return VectorView{vector_bytes.data(), vector_dim, vector_dtype}; }
};

/// Interface every concrete replacement policy implements; see the file
/// overview above for the full contract, candidate lifecycle, and threading
/// model. Mirrors SchedulingPolicy's shape (core/scheduling_policy.hpp):
/// RegionManager owns one concrete instance via std::unique_ptr, defaulted
/// to FifoReplacementPolicy when none is injected.
class ReplacementPolicy {
 public:
	virtual ~ReplacementPolicy() = default;

	/// Consumer side of RegionManager's MPSC intake queue: takes ownership of
	/// one candidate. Called only by the Coordinator thread, once per
	/// candidate; RegionManager keeps no copy of `candidate` after this call
	/// returns.
	virtual void enqueueCandidate(PromotionCandidate candidate) = 0;

	/// `anchor_id` no longer depends on any Region (releaseAnchor() or
	/// evictAnchorNow()); called immediately, never deferred to a trigger.
	/// Must purge `anchor_id` from every structure the policy maintains --
	/// pending candidates and any eviction-ordering state alike. No-op if
	/// the policy isn't tracking `anchor_id`.
	virtual void onAnchorEvicted(VectorId anchor_id) = 0;

	/// `anchor_id` was a dependent of a Region a traversal actually accessed
	/// (see RegionManager::recordTraversal()) -- a pure hotness signal a
	/// policy may ignore (FifoReplacementPolicy does). Runs synchronously on
	/// the traversing thread, not funneled through the Coordinator. May fire
	/// more than once per logical traversal across separate calls, so a
	/// frequency-sensitive policy must dedupe itself.
	virtual void onAnchorTouched(VectorId anchor_id) = 0;

	/// Called once per Coordinator wakeup, before touching pending/resident
	/// state, to decide whether to run a relocation batch this tick.
	/// Returning false costs nothing -- the Coordinator just goes back to
	/// sleep. Bypassed when force-woken by waitIdle()/shutdown().
	virtual bool onRelocationTrigger() = 0;

	/// True if any admitted-but-not-yet-promoted candidate is still sitting
	/// in the policy's own storage. Used by RegionManager::waitIdle() to know
	/// whether an empty intake queue actually means "nothing left to do" --
	/// a candidate can sit here across ticks even with the queue drained.
	virtual bool hasPendingCandidates() const = 0;

	/// Pops the next candidate to act on this round, from the policy's own
	/// storage; called repeatedly until it returns nullopt (implementations
	/// may silently drop stale candidates without ever returning them).
	/// Expected to record the anchor_id into eviction-order state right here
	/// -- there is no separate "promotion succeeded" confirmation call, so a
	/// candidate later found NotEligible by RegionManager::make() still
	/// counts as evictable until onAnchorEvicted() removes it (a wasted
	/// eviction, not a correctness bug -- capacity retries still terminate).
	virtual std::optional<PromotionCandidate> selectNextPromotionCandidate() = 0;

	/// Chooses the next Anchor to reclaim, excluding `excluded` (the Anchor
	/// currently being promoted -- never select the thing being promoted).
	/// Returns nullopt if nothing is eligible to evict.
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

/// LRU (Least-Recently-Used) variant: *admission* (selectNextPromotionCandidate()
/// offering freshly-enqueued candidates) still happens in plain FIFO arrival
/// order -- fairness among candidates nobody has resident state for yet is
/// orthogonal to eviction policy. What differs from FifoReplacementPolicy is
/// *eviction* order: instead of "whichever Anchor was granted its first
/// Region longest ago", this reclaims whichever currently-resident Anchor
/// was *least recently touched by an actual query* -- an Anchor promoted
/// long ago but still queried repeatedly stays resident; one promoted a
/// moment ago but never touched again is the first evicted. This is what
/// onAnchorTouched() (a no-op for FIFO) is for -- see the base interface's
/// own doc comment for the traversal-hotness signal this responds to.
///
/// Implementation: the standard O(1) LRU technique -- a doubly-linked list
/// (`lru_order_`) ordered least- to most-recently-used, plus a hash map of
/// list iterators (`lru_position_`) for O(1) lookup/removal/move-to-back.
/// Becoming resident (selectNextPromotionCandidate() granting a candidate
/// for the first time) counts as a "use" and inserts at the most-recently-
/// used (back) end; onAnchorTouched() splices an already-tracked Anchor
/// there too; selectNextEvictionCandidate() scans from the least-recently-
/// used (front) end, mirroring FifoReplacementPolicy's linear scan of
/// promoted_order_ (skipping `excluded`) -- only called during a capacity
/// retry, not a hot path, so the linear skip-scan is not a concern here
/// either.
class LruReplacementPolicy final : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override;
	void onAnchorEvicted(VectorId anchor_id) override;

	/// Unlike FIFO's no-op: moves `anchor_id` to the most-recently-used end
	/// of lru_order_ if it's currently tracked (i.e. already resident -- see
	/// the class doc comment for why an untracked Anchor here is expected,
	/// not an error: onAnchorTouched() only ever fires for Anchors that
	/// already have a Region dependency).
	void onAnchorTouched(VectorId anchor_id) override;

	bool onRelocationTrigger() override;
	bool hasPendingCandidates() const override;
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override;
	std::optional<VectorId> selectNextEvictionCandidate(VectorId excluded) override;

 private:
	mutable std::mutex mutex_;
	std::deque<PromotionCandidate> pending_candidates_;  // admitted, not yet selected -- oldest-first
	std::list<VectorId> lru_order_;                      // front = least-recently-used, back = most-recently-used
	std::unordered_map<VectorId, std::list<VectorId>::iterator> lru_position_;  // O(1) lookup into lru_order_
};

/// LFU (Least-Frequently-Used) variant: reclaims whichever currently-
/// resident Anchor has been touched the *fewest* times, irrespective of
/// *when* -- an Anchor queried 100 times then left alone for an hour still
/// outranks one queried only twice a second ago. Complements LRU (recency)
/// with frequency; the two make different mistakes on different traffic
/// shapes (LFU is immune to LRU's "one scan of never-repeated queries evicts
/// everything hot" weakness, but is slow to let go of an old favorite that
/// has genuinely gone cold -- see 2Q below for a policy that targets that
/// specific LRU weakness differently).
///
/// Implementation: the frequency-bucket technique (each distinct touch
/// count maps to its own bucket of Anchors at that count), kept in a
/// `std::map<uint64_t, std::list<VectorId>>` ordered by ascending
/// frequency -- `freq_buckets_.begin()` is always the current minimum,
/// so no separate "current minimum frequency" bookkeeping is needed
/// (unlike the classic O(1)-LFU paper's hand-rolled frequency-node list,
/// which exists purely to avoid this map's O(log F) factor, F = number of
/// *distinct* frequency values currently in use -- not the number of
/// Anchors. F stays small in any realistic workload -- bounded by how many
/// times a single Anchor gets touched before being evicted or going cold --
/// so this stays close to O(1) in practice without the extra implementation
/// risk of a hand-rolled structure). onAnchorTouched() moves an Anchor from
/// its current bucket to bucket[freq+1] (O(1) list splice once the bucket
/// is located); becoming resident starts an Anchor at bucket[1] (an
/// implicit first "use", mirroring LRU/Clock's own treatment of grant-time).
/// selectNextEvictionCandidate() walks buckets in ascending frequency order
/// (cheap: only ever as many buckets as distinct frequencies in play, and
/// stops at the first non-excluded match) -- ties within a bucket break
/// FIFO (whichever reached that frequency first), a simpler tie-break than
/// LRU's but still deterministic.
class LfuReplacementPolicy final : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override;
	void onAnchorEvicted(VectorId anchor_id) override;

	/// Unlike FIFO's no-op: bumps `anchor_id`'s touch count by one (moving it
	/// to the next frequency bucket) if it's currently tracked -- see the
	/// class doc comment for why an untracked Anchor here is expected, not an
	/// error.
	void onAnchorTouched(VectorId anchor_id) override;

	bool onRelocationTrigger() override;
	bool hasPendingCandidates() const override;
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override;
	std::optional<VectorId> selectNextEvictionCandidate(VectorId excluded) override;

 private:
	struct TrackedEntry {
		std::uint64_t freq = 0;
		std::list<VectorId>::iterator pos;  // this Anchor's own position within freq_buckets_[freq]
	};

	mutable std::mutex mutex_;
	std::deque<PromotionCandidate> pending_candidates_;  // admitted, not yet selected -- oldest-first
	std::map<std::uint64_t, std::list<VectorId>> freq_buckets_;  // ascending frequency -> Anchors at that frequency
	std::unordered_map<VectorId, TrackedEntry> tracked_;         // O(1) lookup into freq_buckets_
};

/// Clock (Second-Chance) variant: an O(1)-touch approximation of LRU,
/// classic in OS page-replacement for exactly the reason it fits here too --
/// LRU's onAnchorTouched() must relink a node in a doubly-linked list on
/// *every* touch (a handful of pointer writes under the lock); Clock's
/// touch is a single bool store to an already-located slot -- nothing to
/// reorder, so touch's lock-held work is the cheapest of every policy here
/// that actually uses the hotness signal (only FIFO's true no-op touch is
/// cheaper, and that one ignores the signal entirely).
///
/// Implementation: a circular buffer (`ring_`, a `std::vector` of
/// {anchor_id, referenced} pairs) plus a hash map of ring indices
/// (`position_`) for O(1) lookup, and a sweeping `hand_` index.
/// onAnchorTouched() just sets the located slot's `referenced` bit.
/// selectNextEvictionCandidate() walks forward from `hand_`: an Anchor with
/// `referenced == true` is given a second chance (bit cleared, hand
/// advances past it) instead of being evicted immediately; the first
/// non-excluded Anchor found with `referenced == false` is the victim
/// (bounded by two full sweeps of the ring -- every entry can only be
/// spared once before its bit is already clear on the second pass).
/// Removal (onAnchorEvicted()) is O(1) via swap-with-last-element-then-
/// pop_back, which is why `ring_` doesn't preserve insertion order --
/// Clock never needed that order in the first place, only the sweep
/// position, which `hand_` tracks independently and is simply reset to 0
/// after any removal (losing exact sweep position on the rare removal path
/// is a standard, correctness-preserving simplification -- it does not
/// affect which Anchor is *eventually* evicted, only how many extra no-op
/// steps the next sweep takes to get there).
class ClockReplacementPolicy final : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override;
	void onAnchorEvicted(VectorId anchor_id) override;

	/// Unlike FIFO's no-op: sets `anchor_id`'s reference bit if it's
	/// currently tracked -- see the class doc comment for why this is the
	/// cheapest possible non-trivial touch (one lookup, one bool store, no
	/// container reordering).
	void onAnchorTouched(VectorId anchor_id) override;

	bool onRelocationTrigger() override;
	bool hasPendingCandidates() const override;
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override;
	std::optional<VectorId> selectNextEvictionCandidate(VectorId excluded) override;

 private:
	struct ClockEntry {
		VectorId anchor_id = 0;
		bool referenced = false;
	};

	mutable std::mutex mutex_;
	std::deque<PromotionCandidate> pending_candidates_;  // admitted, not yet selected -- oldest-first
	std::vector<ClockEntry> ring_;                       // unordered -- see class doc comment
	std::unordered_map<VectorId, std::size_t> position_;  // O(1) lookup into ring_
	std::size_t hand_ = 0;                                // next ring_ index the sweep resumes from
};

/// 2Q variant (Johnson & Shasha 1994): targets a specific weakness plain LRU
/// (and, to a lesser extent, LFU) has -- a burst of once-only queries (a
/// "scan") can evict Anchors that were genuinely, repeatedly hot, because
/// LRU treats "just placed" identically to "long proven". 2Q fixes this by
/// splitting resident Anchors into two queues: `a1in_`, a plain FIFO for
/// Anchors seen for the first time, and `am_`, an LRU queue for Anchors
/// that have *proven* themselves by being touched again while still in
/// `a1in_`. A third, data-less "ghost" queue (`a1out_`, bounded, holds ids
/// only) remembers Anchors recently evicted from `a1in_` -- if one of those
/// ids is granted a fresh promotion again, it skips `a1in_` entirely and
/// goes straight into the protected `am_`, since a returning visitor has
/// already demonstrated it isn't just scan noise.
///
/// Eviction always prefers `a1in_` (oldest first) over `am_` (least-
/// recently-used first): a first-timer is always a cheaper, safer sacrifice
/// than something already proven -- this is the mechanism that actually
/// protects `am_` from scan pollution, and (unlike the classic
/// paper/production 2Q implementations) needs no externally-supplied
/// capacity/byte-budget split between the two queues to work, since
/// ReplacementPolicy is never told the GPU budget in the first place (see
/// core/region_manager.hpp) -- `a1in_` is simply drained first whenever
/// eviction is needed at all, rather than only once it exceeds some target
/// fraction.
///
/// Lock-cost note: onAnchorTouched()/selectNextPromotionCandidate() each do
/// up to two map lookups (checking `a1in_` then `am_`) plus a list
/// splice/insert -- strictly more work under the lock than plain LRU's
/// single lookup+splice, in exchange for the scan-resistance above. Of the
/// four policies in this file, this is the most expensive per touch; Clock
/// is the cheapest, LFU in between.
///
/// `onAnchorEvicted()` cannot distinguish *why* an Anchor left residency
/// (RegionManager's capacity-driven evictAnchorNow() and delete-driven
/// releaseAnchor() both funnel through the same notification -- see the
/// base interface's own doc comment) -- only an `a1in_` departure is ever
/// fed into `a1out_`'s ghost memory (an `am_` departure is not: it already
/// proved itself once, so a future revisit starting fresh through `a1in_`
/// is the standard, intended 2Q behavior, not a gap). A deleted Anchor
/// therefore still leaves a short-lived ghost entry behind; the only
/// consequence if that exact VectorId is reused by an unrelated later
/// insert (see PromotionCandidate's own epoch doc comment for that race) is
/// a placement-heuristic mistake (straight into `am_` instead of `a1in_`),
/// never a correctness issue -- eviction order is a preference, not a
/// safety property.
class TwoQReplacementPolicy final : public ReplacementPolicy {
 public:
	/// `ghost_capacity` bounds `a1out_`'s size (oldest ghost entry is dropped
	/// once exceeded) -- a small, fixed amount of id-only bookkeeping,
	/// independent of how large GPU residency itself is allowed to grow.
	explicit TwoQReplacementPolicy(std::size_t ghost_capacity = 256);

	void enqueueCandidate(PromotionCandidate candidate) override;
	void onAnchorEvicted(VectorId anchor_id) override;
	void onAnchorTouched(VectorId anchor_id) override;
	bool onRelocationTrigger() override;
	bool hasPendingCandidates() const override;
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override;
	std::optional<VectorId> selectNextEvictionCandidate(VectorId excluded) override;

 private:
	std::size_t ghost_capacity_;

	mutable std::mutex mutex_;
	std::deque<PromotionCandidate> pending_candidates_;  // admitted, not yet selected -- oldest-first

	std::list<VectorId> a1in_;  // first-timers, FIFO, oldest-first
	std::unordered_map<VectorId, std::list<VectorId>::iterator> a1in_position_;

	std::list<VectorId> am_;  // proven-hot, LRU, least-to-most-recently-used
	std::unordered_map<VectorId, std::list<VectorId>::iterator> am_position_;

	std::list<VectorId> a1out_;  // ghost FIFO: ids only, no PromotionCandidate data
	std::unordered_map<VectorId, std::list<VectorId>::iterator> a1out_position_;
};

}  // namespace arachne
