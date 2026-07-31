#include "gpu/compaction_policy.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "gpu/device_context.hpp"

namespace {

using arachne::gpu::AllocationPolicy;
using arachne::gpu::DeviceContext;
using arachne::gpu::MovableBlock;
using arachne::gpu::NoCompactionPolicy;
using arachne::gpu::TargetedCompactionPolicy;
using arachne::gpu::UnitPoolArena;

constexpr std::size_t kUnitBytes = 256;

// Validates TargetedCompactionPolicy's plan() search in isolation: given a
// synthetic arena layout (holes and movable blocks placed directly via
// allocateBestFit()/free()/claim(), not through DeviceRegionPool), does it
// find the cheapest relocation set that opens up `required_units`, or
// correctly report infeasible? NoCompactionPolicy's always-decline contract
// is covered too. DeviceRegionPool's own execution/re-validation of a
// returned Plan is out of scope here -- see device_region_pool_test.cpp.
class CompactionPolicyTest : public ::testing::Test {
 protected:
	DeviceContext device_{/*device_id=*/0, AllocationPolicy::Normal};
};

TEST_F(CompactionPolicyTest, NoCompactionPolicyAlwaysDeclines) {
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	NoCompactionPolicy policy;

	auto plan = policy.plan(arena, {}, /*required_units=*/1);
	EXPECT_FALSE(plan.feasible);
	EXPECT_TRUE(plan.moves.empty());
}

TEST_F(CompactionPolicyTest, TargetedPolicyIsFeasibleWithZeroMovesWhenAlreadySatisfiable) {
	// The whole arena is one free extent -- any required_units within it
	// needs no relocation at all.
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	TargetedCompactionPolicy policy;

	auto plan = policy.plan(arena, /*movable=*/{}, /*required_units=*/4);
	EXPECT_TRUE(plan.feasible);
	EXPECT_TRUE(plan.moves.empty());
}

TEST_F(CompactionPolicyTest, TargetedPolicyReturnsInfeasibleWhenGenuinelyOutOfSpace) {
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	TargetedCompactionPolicy policy;

	// Nothing is even occupied -- total capacity itself is short of the
	// request, so no relocation plan (regardless of budget) can help.
	auto plan = policy.plan(arena, /*movable=*/{}, /*required_units=*/11);
	EXPECT_FALSE(plan.feasible);
}

TEST_F(CompactionPolicyTest, TargetedPolicyRelocatesTheSingleBlockBetweenTwoHoles) {
	// [0,2) hole, [2,4) movable, [4,6) hole, [6,8) movable, [8,10) hole --
	// no single hole fits a 4-unit request, but relocating either movable
	// block into the *other* hole grows a hole to exactly 4.
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	std::vector<UnitPoolArena::UnitRange> pieces;
	for (int i = 0; i < 5; ++i) {
		auto piece = arena.allocateBestFit(2);
		ASSERT_TRUE(piece.has_value());
		pieces.push_back(*piece);
	}
	arena.free(pieces[0]);  // [0,2)
	arena.free(pieces[2]);  // [4,6)
	arena.free(pieces[4]);  // [8,10)
	// pieces[1] = [2,4), pieces[3] = [6,8) remain occupied -- these are the
	// two candidate movable blocks.
	ASSERT_EQ(arena.totalFreeUnits(), 6u);
	ASSERT_EQ(arena.largestFreeExtent(), 2u);

	std::vector<MovableBlock> movable{MovableBlock{101, pieces[1]}, MovableBlock{103, pieces[3]}};

	TargetedCompactionPolicy policy;
	auto plan = policy.plan(arena, movable, /*required_units=*/4);

	ASSERT_TRUE(plan.feasible);
	ASSERT_EQ(plan.moves.size(), 1u);
	// Both candidates (growing [0,2) by moving block 101, or growing [4,6)
	// by moving block 103) cost the same (2 moved units) -- ties break by
	// lowest extent start address, so the [0,2) candidate (moving 101) wins.
	EXPECT_EQ(plan.moves[0].block_id, 101u);
	EXPECT_EQ(plan.moves[0].from.start_unit, 2u);
	EXPECT_EQ(plan.moves[0].to.start_unit, 4u);  // the OTHER hole, not the one being grown
	EXPECT_EQ(plan.moves[0].to.unit_count, 2u);
}

TEST_F(CompactionPolicyTest, TargetedPolicyAccumulatesMultipleConsecutiveMovableBlocks) {
	// [0,2) hole, A=[2,4) movable, B=[4,7) movable, [7,12) hole -- neither
	// hole alone reaches 6 units, nor does either single block; only
	// relocating *both* A and B clears the needed space, and the tail hole
	// is exactly big enough to receive both.
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 12);
	UnitPoolArena::UnitRange a{2, 2};
	UnitPoolArena::UnitRange b{4, 3};
	arena.claim(a);
	arena.claim(b);
	ASSERT_EQ(arena.totalFreeUnits(), 7u);  // [0,2) + [7,12)
	ASSERT_EQ(arena.largestFreeExtent(), 5u);

	std::vector<MovableBlock> movable{MovableBlock{201, a}, MovableBlock{202, b}};
	TargetedCompactionPolicy policy;
	auto plan = policy.plan(arena, movable, /*required_units=*/6);

	ASSERT_TRUE(plan.feasible);
	ASSERT_EQ(plan.moves.size(), 2u);
	EXPECT_EQ(plan.moves[0].block_id, 201u);
	EXPECT_EQ(plan.moves[1].block_id, 202u);
	// Both destinations must land inside the tail hole ([7,12)), and must not
	// overlap each other.
	for (const auto& move : plan.moves) {
		EXPECT_GE(move.to.start_unit, 7u);
		EXPECT_LE(move.to.end(), 12u);
	}
	EXPECT_NE(plan.moves[0].to.start_unit, plan.moves[1].to.start_unit);
}

TEST_F(CompactionPolicyTest, TargetedPolicyRespectsMaxMovedBlocksBudget) {
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	std::vector<UnitPoolArena::UnitRange> pieces;
	for (int i = 0; i < 5; ++i) {
		auto piece = arena.allocateBestFit(2);
		ASSERT_TRUE(piece.has_value());
		pieces.push_back(*piece);
	}
	arena.free(pieces[0]);
	arena.free(pieces[2]);
	arena.free(pieces[4]);
	std::vector<MovableBlock> movable{MovableBlock{101, pieces[1]}, MovableBlock{103, pieces[3]}};

	TargetedCompactionPolicy::Budget budget;
	budget.max_moved_blocks = 0;  // not even one relocation permitted
	TargetedCompactionPolicy policy(budget);

	auto plan = policy.plan(arena, movable, /*required_units=*/4);
	EXPECT_FALSE(plan.feasible);
}

TEST_F(CompactionPolicyTest, TargetedPolicyRespectsMaxMovedRatioBudget) {
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	std::vector<UnitPoolArena::UnitRange> pieces;
	for (int i = 0; i < 5; ++i) {
		auto piece = arena.allocateBestFit(2);
		ASSERT_TRUE(piece.has_value());
		pieces.push_back(*piece);
	}
	arena.free(pieces[0]);
	arena.free(pieces[2]);
	arena.free(pieces[4]);
	std::vector<MovableBlock> movable{MovableBlock{101, pieces[1]}, MovableBlock{103, pieces[3]}};

	TargetedCompactionPolicy::Budget budget;
	budget.max_moved_ratio = 0.1;  // 4 * 0.1 == 0 (truncated) moved units allowed
	TargetedCompactionPolicy policy(budget);

	auto plan = policy.plan(arena, movable, /*required_units=*/4);
	EXPECT_FALSE(plan.feasible);
}

TEST_F(CompactionPolicyTest, TargetedPolicyIgnoresBlocksNotOfferedAsMovable) {
	// A block that's currently pinned (e.g. an outstanding Lease) must never
	// be offered as movable by the caller -- from the policy's perspective
	// that space is an opaque wall, so it correctly reports infeasible
	// rather than moving something it was never told about.
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10);
	std::vector<UnitPoolArena::UnitRange> pieces;
	for (int i = 0; i < 5; ++i) {
		auto piece = arena.allocateBestFit(2);
		ASSERT_TRUE(piece.has_value());
		pieces.push_back(*piece);
	}
	arena.free(pieces[0]);
	arena.free(pieces[2]);
	arena.free(pieces[4]);

	TargetedCompactionPolicy policy;
	auto plan = policy.plan(arena, /*movable=*/{}, /*required_units=*/4);  // neither occupied block offered
	EXPECT_FALSE(plan.feasible);
}

}  // namespace
