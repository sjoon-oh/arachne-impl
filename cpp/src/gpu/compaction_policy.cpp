#include "gpu/compaction_policy.hpp"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>

namespace arachne::gpu {

CompactionPolicy::Plan NoCompactionPolicy::plan(const UnitPoolArena& /*arena*/,
																								 const std::vector<MovableBlock>& /*movable*/,
																								 std::uint64_t /*required_units*/) {
	return Plan{};  // moves empty, feasible=false -- see class doc comment
}

TargetedCompactionPolicy::TargetedCompactionPolicy(Budget budget) : budget_(budget) {}

namespace {

// One entry in the merged, address-ordered view plan() walks: either a free
// extent or a currently-movable (unpinned) live block. Anything else
// (pinned, or simply not part of `movable`) is deliberately absent -- a gap
// between two consecutive Segments in address order is how a hard wall
// (something growth can never cross) shows up, without this code needing to
// know anything about what's actually occupying it.
struct Segment {
	UnitPoolArena::UnitRange range;
	bool is_free = false;
	std::uint64_t block_id = 0;  // meaningful only if !is_free
};

// One free extent's candidate rightward-growth outcome: starting from a
// free extent at `extent_start`, relocate `block_ids` (in address order,
// each touching the next) until the union reaches the caller's
// required_units or budget runs out.
struct Candidate {
	std::uint64_t extent_start = 0;
	std::vector<std::uint64_t> block_ids;
	std::uint64_t moved_units = 0;
};

}  // namespace

// Two-phase search: grow candidates, then try to place their displaced
// blocks. `free_extents` and `movable` are merged into one address-ordered
// timeline of Segments (free extent or unpinned block); everything else
// (pinned blocks, or gaps belonging to neither list) is invisible to this
// walk and simply acts as a hard wall growth can't cross:
//
//   address ->
//   [   F0   ][ B1 ][ B2 ][   F1   ][ /////P///// ][ B3 ][   F2   ]
//    `------- grow rightward ------'   (wall: not       `- B3 is its
//     from F0 through B1, B2 while      in `movable`,      own candidate
//     budget/required_units allow        skipped)          starting at F2
//
// Phase 1 walks every free extent and greedily merges the movable blocks
// immediately to its right (address-contiguous, within budget) until
// required_units is reached -- one Candidate per free extent that gets
// there. This deliberately only considers "grow one extent rightward"
// plans, not arbitrary non-adjacent relocations (that's an NP-hard-flavored
// packing problem); it's a bounded, terminating approximation that only
// proposes the moves a human would recognize as obviously cheap.
//
// Phase 2 takes candidates cheapest-first (fewest moved units, see the
// comment at the sort below) and tries to give each of its displaced
// blocks a distinct destination among the *other* free extents, via a
// locally-simulated best-fit -- simulated so a failed candidate leaves
// `arena` untouched and the next cheapest one can still be tried. The first
// candidate whose blocks all get a destination wins; if none do, the plan
// is infeasible.
CompactionPolicy::Plan TargetedCompactionPolicy::plan(const UnitPoolArena& arena,
																											 const std::vector<MovableBlock>& movable,
																											 std::uint64_t required_units) {
	Plan result;
	if (required_units == 0) {
		result.feasible = true;
		return result;
	}

	std::vector<UnitPoolArena::UnitRange> free_extents = arena.freeExtentsByAddress();

	std::uint64_t total_free = 0;
	for (const UnitPoolArena::UnitRange& extent : free_extents) total_free += extent.unit_count;
	if (total_free < required_units) return result;  // genuine OOM -- no relocation plan can help

	std::uint64_t moved_budget_units =
			static_cast<std::uint64_t>(static_cast<double>(required_units) * budget_.max_moved_ratio);

	std::vector<Segment> segments;
	segments.reserve(free_extents.size() + movable.size());
	for (const UnitPoolArena::UnitRange& extent : free_extents) segments.push_back(Segment{extent, true, 0});
	for (const MovableBlock& block : movable) segments.push_back(Segment{block.range, false, block.id});
	std::sort(segments.begin(), segments.end(),
						[](const Segment& a, const Segment& b) { return a.range.start_unit < b.range.start_unit; });

	// Phase 1: grow each free extent rightward (see algorithm overview above).
	std::vector<Candidate> candidates;
	for (std::size_t i = 0; i < segments.size(); ++i) {
		if (!segments[i].is_free) continue;

		Candidate candidate;
		candidate.extent_start = segments[i].range.start_unit;
		std::uint64_t achieved = segments[i].range.unit_count;
		std::uint64_t frontier = segments[i].range.end();

		std::size_t j = i + 1;
		while (achieved < required_units && j < segments.size() && !segments[j].is_free &&
					 segments[j].range.start_unit == frontier &&
					 candidate.moved_units + segments[j].range.unit_count <= moved_budget_units &&
					 candidate.block_ids.size() < budget_.max_moved_blocks) {
			achieved += segments[j].range.unit_count;
			candidate.moved_units += segments[j].range.unit_count;
			candidate.block_ids.push_back(segments[j].block_id);
			frontier = segments[j].range.end();
			++j;
		}

		if (achieved >= required_units) candidates.push_back(std::move(candidate));
	}

	// Cheapest (fewest moved units, then fewest blocks, then leftmost) first
	// -- see class doc comment for why moved units is the right objective.
	std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
		if (a.moved_units != b.moved_units) return a.moved_units < b.moved_units;
		if (a.block_ids.size() != b.block_ids.size()) return a.block_ids.size() < b.block_ids.size();
		return a.extent_start < b.extent_start;
	});

	std::unordered_map<std::uint64_t, UnitPoolArena::UnitRange> range_of;
	range_of.reserve(movable.size());
	for (const MovableBlock& block : movable) range_of.emplace(block.id, block.range);

	// Phase 2: assign destinations, cheapest candidate first (see overview above).
	for (const Candidate& candidate : candidates) {
		std::multimap<std::uint64_t, std::uint64_t> sim_free_by_size;  // unit_count -> start_unit
		for (const UnitPoolArena::UnitRange& extent : free_extents) {
			if (extent.start_unit == candidate.extent_start) continue;
			sim_free_by_size.emplace(extent.unit_count, extent.start_unit);
		}

		std::vector<Move> moves;
		bool ok = true;
		for (std::uint64_t block_id : candidate.block_ids) {
			UnitPoolArena::UnitRange from = range_of.at(block_id);
			auto it = sim_free_by_size.lower_bound(from.unit_count);
			if (it == sim_free_by_size.end()) {
				ok = false;
				break;
			}
			UnitPoolArena::UnitRange to{it->second, from.unit_count};
			moves.push_back(Move{block_id, from, to});

			std::uint64_t remaining = it->first - from.unit_count;
			std::uint64_t remaining_start = it->second + from.unit_count;
			sim_free_by_size.erase(it);
			if (remaining > 0) sim_free_by_size.emplace(remaining, remaining_start);
		}

		if (ok) {
			result.moves = std::move(moves);
			result.feasible = true;
			return result;
		}
	}

	return result;  // every candidate reaching required_units failed destination assignment
}

}  // namespace arachne::gpu
