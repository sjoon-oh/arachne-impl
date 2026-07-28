#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <unordered_set>

#include "types.hpp"

namespace arachne {

/// Pluggable Anchor-level Stitch replacement policy (Quick Summary design
/// point 4): decides which Anchor's Stitches to reclaim when Controller
/// needs to make room for a Promotion and nothing is already free. Always
/// operates on Anchor ids, never on individual Stitches/Regions -- an
/// Anchor and every Stitch AnchorManager currently holds for it (see
/// core/anchor_manager.hpp) are the unit of locality this policy reasons
/// about, per the Anchor-centric residency design (replacement is about
/// which *Anchor* has gone cold, not which address range looks sparse).
///
/// Mirrors SchedulingPolicy's shape (core/scheduling_policy.hpp): a pure
/// interface here, concrete strategies (FifoReplacementPolicy today; LRU or
/// a GPU-aware hotness/latency/transfer-cost score later) as separate
/// classes, Controller only ever calling through this interface -- and
/// Controller owns the concrete instance the same way OpScheduler owns a
/// SchedulingPolicy (std::unique_ptr, defaulted to Fifo* when none is
/// injected).
///
/// A policy is trusted to stay in sync purely from the onStitchAdded()/
/// onAnchorEvicted() notifications Controller sends it -- it never reaches
/// into AnchorManager itself to cross-check. Every method must be
/// thread-safe: Controller is called concurrently the same way
/// AnchorManager is.
class ReplacementPolicy {
 public:
	virtual ~ReplacementPolicy() = default;

	/// Notifies the policy that `anchor_id` now holds at least one Stitch
	/// (Controller calls this from make() right after a Stitch is
	/// successfully added). No-op if the policy is already tracking
	/// `anchor_id` -- gaining a second/third Stitch doesn't change its
	/// standing under a policy that orders purely by "first granted".
	virtual void onStitchAdded(VectorId anchor_id) = 0;

	/// Notifies the policy that `anchor_id` no longer holds any Stitch
	/// (Controller calls this from evictAnchor(), after every Stitch on that
	/// Anchor has actually been reclaimed). No-op if the policy isn't
	/// tracking `anchor_id`.
	virtual void onAnchorEvicted(VectorId anchor_id) = 0;

	/// Chooses the next Anchor to reclaim to make room for a Promotion,
	/// excluding `excluded` (the Anchor currently being promoted -- a policy
	/// must never select the thing it's making room for). Returns nullopt if
	/// there is nothing eligible to evict.
	virtual std::optional<VectorId> selectEvictionCandidate(VectorId excluded) const = 0;
};

/// Default policy: reclaims whichever currently-tracked Anchor was granted
/// its first Stitch longest ago, irrespective of any Anchor-query hotness
/// signal since. Deliberately the simplest possible strategy -- it stands
/// up the Eviction -> (optional) Compaction -> Promotion pipeline's
/// skeleton (see gpu/stitch_pool.hpp's compact() doc comment for the GPU
/// memory half of that pipeline) before a real Anchor-query-aware scoring
/// policy replaces it.
class FifoReplacementPolicy final : public ReplacementPolicy {
 public:
	void onStitchAdded(VectorId anchor_id) override;
	void onAnchorEvicted(VectorId anchor_id) override;
	std::optional<VectorId> selectEvictionCandidate(VectorId excluded) const override;

 private:
	mutable std::mutex mutex_;
	std::deque<VectorId> order_;  // oldest-first; front is the next candidate
	std::unordered_set<VectorId> tracked_;
};

}  // namespace arachne
