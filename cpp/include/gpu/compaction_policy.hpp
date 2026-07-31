#pragma once

#include <cstdint>
#include <vector>

#include "gpu/unit_pool_arena.hpp"

namespace arachne::gpu {

/// ---------------------------------------------------------------------
/// Compaction subsystem overview
/// ---------------------------------------------------------------------
///
/// Generic allocators (RMM's pool_memory_resource included) can do best-fit
/// + coalescing, but structurally cannot solve external fragmentation: the
/// case where totalFreeUnits() >= required but largestFreeExtent() <
/// required (see UnitPoolArena's own doc comment). Closing that gap
/// requires relocating still-*live* allocations, which needs
/// application-level knowledge (which allocations are safe to move right
/// now, and where to) that a generic allocator doesn't have.
/// CompactionPolicy is the injectable strategy for that decision; it is
/// only ever consulted by DeviceRegionPool, and only inside that exact
/// external-fragmentation window -- plan() is never called outside it.
///
/// Split of responsibility (mirrors ReplacementPolicy/RegionManager, see
/// core/replacement_policy.hpp's own class doc comment -- a policy's
/// suggestion is never trusted blindly):
///
///   CompactionPolicy::plan()                DeviceRegionPool (executor)
///   -----------------------------------     -----------------------------
///   Offered a read-only snapshot of         Owns the arena and the
///   movable blocks plus `arena`'s           allocations_ map. Builds the
///   free-extent state. Decides which        MovableBlock snapshot (never
///   blocks to move and where. Must not      including a currently-pinned
///   mutate `arena` or touch GPU memory      allocation -- in_use_count
///   or DeviceRegionPool's own state.        != 0, i.e. an outstanding
///                                           Lease) and executes the
///                                           returned Plan move by move,
///                                           re-validating each Move
///                                           against live state
///                                           immediately before performing
///                                           it (cheap: its own lock has
///                                           been held continuously since
///                                           the snapshot).
///
/// Two concrete policies are provided:
///
///  - NoCompactionPolicy: always declines. Lets a caller opt a Pooled
///    DeviceRegionPool out of D2D relocation entirely, so a fragmentation
///    failure behaves exactly as it would with no compaction mechanism at
///    all (best-fit + coalescing only) -- useful for benchmarking how much
///    a real policy actually buys, or for a deployment that would rather
///    fail an allocation than ever pay a D2D copy cost.
///
///  - TargetedCompactionPolicy: the default Pooled-mode strategy. Bounded,
///    single-hop "targeted compaction" (see cpp/doc/compaction-doc-plan.md's
///    own terminology): grows exactly one existing free extent by
///    relocating the run of movable blocks immediately to its right, until
///    it reaches required_units or its Budget is exhausted. Deliberately
///    does not search the full combinatorial space of relocation plans
///    (moving arbitrary, non-adjacent subsets of blocks to open space) --
///    that's an NP-hard-flavored packing problem in general, and this is a
///    bounded, terminating approximation that only considers the
///    "obviously cheap" single-hop-rightward-run candidates (the doc's Plan
///    A/B/C-style candidates), consistent with the doc's own v1 guidance to
///    avoid unbounded sliding compaction. Among every free extent that can
///    reach required_units within budget this way, the one whose relocated
///    blocks sum to the fewest moved units wins (ties: fewest blocks moved,
///    then lowest starting address, for determinism) -- moved bytes is the
///    dominant real cost (GPU bandwidth + the window a relocated block is
///    unavailable to acquire()), so minimizing it directly is the right
///    objective, not a proxy for one.
///
///    Its Budget (CompactionBudget) bounds how much work one plan() call
///    may propose, so a single fragmented-allocation event can never turn
///    into an unbounded D2D copy storm (see compaction-doc-plan.md section
///    12): max_moved_ratio caps total moved units as a multiple of
///    required_units (e.g. 2.0 allows moving up to 2x the units actually
///    requested); max_moved_blocks caps the move count outright, since many
///    tiny blocks cost more in bookkeeping/event-wait overhead than their
///    byte count alone suggests.

/// One live allocation DeviceRegionPool currently considers safe to
/// relocate -- see the compaction subsystem overview above.
struct MovableBlock {
	std::uint64_t id = 0;
	UnitPoolArena::UnitRange range;
};

/// See the compaction subsystem overview above for the policy/executor
/// split this class is one half of.
class CompactionPolicy {
 public:
	virtual ~CompactionPolicy() = default;

	/// One live allocation to relocate: `from` must exactly match a
	/// MovableBlock::range offered to plan(); `to` must be a subrange of a
	/// free extent `arena` reported at plan() time and must not overlap any
	/// other Move's `to` in the same Plan.
	struct Move {
		std::uint64_t block_id = 0;
		UnitPoolArena::UnitRange from;
		UnitPoolArena::UnitRange to;
	};

	/// `feasible` false means `moves` is always empty. A feasible Plan's
	/// `moves`, executed in order, are expected to leave *some* free extent
	/// >= required_units -- DeviceRegionPool re-runs allocateBestFit()
	/// itself afterward rather than trusting a specific target address.
	struct Plan {
		std::vector<Move> moves;
		bool feasible = false;
	};

	/// `movable` is address-ordered by `range.start_unit`. Must return a
	/// Plan referencing only block ids present in `movable`, with `to`
	/// ranges that are subranges of what `arena.freeExtentsByAddress()`
	/// reported free -- any other Move is an internal-invariant violation
	/// the executor does not tolerate or repair.
	virtual Plan plan(const UnitPoolArena& arena, const std::vector<MovableBlock>& movable,
										 std::uint64_t required_units) = 0;
};

/// Always declines -- see the compaction subsystem overview above.
class NoCompactionPolicy final : public CompactionPolicy {
 public:
	Plan plan(const UnitPoolArena& arena, const std::vector<MovableBlock>& movable,
						std::uint64_t required_units) override;
};

/// Namespace-scope rather than nested in TargetedCompactionPolicy (which
/// aliases it as Budget) because GCC rejects a default constructor argument
/// of a class type whose default member initializers are defined inside
/// that same enclosing class -- see the compaction overview above for what
/// these bounds mean.
struct CompactionBudget {
	double max_moved_ratio = 2.0;
	std::uint64_t max_moved_blocks = 8;
};

/// See the compaction subsystem overview above for the targeted-compaction
/// algorithm and its selection criteria.
class TargetedCompactionPolicy final : public CompactionPolicy {
 public:
	using Budget = CompactionBudget;

	explicit TargetedCompactionPolicy(Budget budget = {});

	Plan plan(const UnitPoolArena& arena, const std::vector<MovableBlock>& movable,
						std::uint64_t required_units) override;

 private:
	Budget budget_;
};

}  // namespace arachne::gpu
