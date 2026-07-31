#include "gpu/unit_pool_arena.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "gpu/device_context.hpp"

namespace {

using arachne::gpu::AllocationPolicy;
using arachne::gpu::DeviceContext;
using arachne::gpu::UnitPoolArena;

constexpr std::size_t kUnitBytes = 256;

// Validates UnitPoolArena's allocation primitives directly (best-fit
// search, free-with-coalescing, claim/relocate) against a real CUDA-backed
// DeviceContext, independent of DeviceRegionPool or any CompactionPolicy.
// Several cases build the same "every-other-piece freed" fragmented layout
// that CompactionPolicy exists to resolve -- see compaction_policy_test.cpp
// for how a policy plans around exactly that shape.

// AllocationPolicy::Normal so DeviceContext doesn't also preallocate its own
// arena -- these UnitPoolArena instances are constructed directly by each
// test, the same way DeviceContext builds its own arenas under Pooled.
class UnitPoolArenaTest : public ::testing::Test {
 protected:
	DeviceContext device_{/*device_id=*/0, AllocationPolicy::Normal};
	UnitPoolArena arena_{device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 10};
};

TEST_F(UnitPoolArenaTest, ReportsUnitAndTotalCapacity) {
	EXPECT_EQ(arena_.unitBytes(), kUnitBytes);
	EXPECT_EQ(arena_.totalUnits(), 10u);
	EXPECT_EQ(arena_.totalFreeUnits(), 10u);
	EXPECT_EQ(arena_.largestFreeExtent(), 10u);
}

TEST_F(UnitPoolArenaTest, RoundsTotalBytesUpToUnitMultiple) {
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, kUnitBytes * 3 + 1);
	EXPECT_EQ(arena.totalUnits(), 4u);  // ceil((3*256 + 1) / 256) == 4
}

TEST_F(UnitPoolArenaTest, ZeroTotalBytesYieldsAnEmptyArena) {
	UnitPoolArena arena(device_.dataResource(), device_.resources(), kUnitBytes, 0);
	EXPECT_EQ(arena.totalUnits(), 0u);
	EXPECT_EQ(arena.totalFreeUnits(), 0u);
	EXPECT_FALSE(arena.allocateBestFit(1).has_value());
}

TEST_F(UnitPoolArenaTest, ZeroUnitBytesThrows) {
	EXPECT_THROW(UnitPoolArena(device_.dataResource(), device_.resources(), 0, kUnitBytes), std::invalid_argument);
}

TEST_F(UnitPoolArenaTest, AllocateZeroUnitsIsANoopRange) {
	auto range = arena_.allocateBestFit(0);
	ASSERT_TRUE(range.has_value());
	EXPECT_EQ(range->unit_count, 0u);
	EXPECT_EQ(arena_.totalFreeUnits(), 10u);  // untouched
}

TEST_F(UnitPoolArenaTest, AllocateBestFitClaimsExactlyRequiredUnitsAndLeavesRemainder) {
	auto range = arena_.allocateBestFit(4);
	ASSERT_TRUE(range.has_value());
	EXPECT_EQ(range->start_unit, 0u);
	EXPECT_EQ(range->unit_count, 4u);
	EXPECT_EQ(arena_.totalFreeUnits(), 6u);
	EXPECT_EQ(arena_.largestFreeExtent(), 6u);
}

TEST_F(UnitPoolArenaTest, AllocateBestFitPicksSmallestSufficientExtent) {
	// [0,3) hole, [3,5) wall (kept occupied), [5,10) hole -- a 2-unit request
	// must land in the 3-unit hole, not the 5-unit one.
	auto a = arena_.allocateBestFit(3);  // [0,3)
	auto wall = arena_.allocateBestFit(2);  // [3,5)
	auto c = arena_.allocateBestFit(5);  // [5,10)
	ASSERT_TRUE(a.has_value() && wall.has_value() && c.has_value());
	arena_.free(*a);
	arena_.free(*c);
	EXPECT_EQ(arena_.totalFreeUnits(), 8u);
	EXPECT_EQ(arena_.largestFreeExtent(), 5u);

	auto best = arena_.allocateBestFit(2);
	ASSERT_TRUE(best.has_value());
	EXPECT_EQ(best->start_unit, 0u);  // the 3-unit hole, not the 5-unit one
	EXPECT_EQ(best->unit_count, 2u);
}

TEST_F(UnitPoolArenaTest, AllocateBestFitReturnsNulloptWhenNoSingleExtentFitsDespiteEnoughAggregateFreeSpace) {
	// Five 2-unit pieces filling the whole arena; free every other one so no
	// two holes are adjacent -- exactly the classic external-fragmentation
	// shape CompactionPolicy exists to resolve.
	std::vector<UnitPoolArena::UnitRange> pieces;
	for (int i = 0; i < 5; ++i) {
		auto piece = arena_.allocateBestFit(2);
		ASSERT_TRUE(piece.has_value());
		pieces.push_back(*piece);
	}
	arena_.free(pieces[0]);
	arena_.free(pieces[2]);
	arena_.free(pieces[4]);

	EXPECT_EQ(arena_.totalFreeUnits(), 6u);       // 3 holes of 2 units each
	EXPECT_EQ(arena_.largestFreeExtent(), 2u);    // but no single one is big enough
	EXPECT_FALSE(arena_.allocateBestFit(4).has_value());
}

TEST_F(UnitPoolArenaTest, FreeCoalescesWithBothNeighborsAtOnce) {
	auto a = arena_.allocateBestFit(3);  // [0,3)
	auto b = arena_.allocateBestFit(2);  // [3,5)
	auto c = arena_.allocateBestFit(3);  // [5,8)
	auto d = arena_.allocateBestFit(2);  // [8,10)
	ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value() && d.has_value());

	arena_.free(*a);  // [0,3) free, no neighbors
	arena_.free(*c);  // [5,8) free, no neighbors (b still occupies [3,5))
	EXPECT_EQ(arena_.largestFreeExtent(), 3u);
	EXPECT_EQ(arena_.totalFreeUnits(), 6u);

	arena_.free(*b);  // [3,5) frees -- merges with BOTH [0,3) and [5,8) at once
	EXPECT_EQ(arena_.largestFreeExtent(), 8u);
	EXPECT_EQ(arena_.totalFreeUnits(), 8u);  // d ([8,10)) is still live
}

TEST_F(UnitPoolArenaTest, ClaimThrowsWhenRangeIsNotEntirelyFree) {
	auto occupied = arena_.allocateBestFit(4);
	ASSERT_TRUE(occupied.has_value());
	EXPECT_THROW(arena_.claim(*occupied), std::logic_error);

	// Straddling a free/occupied boundary is likewise not entirely free.
	EXPECT_THROW(arena_.claim(UnitPoolArena::UnitRange{3, 4}), std::logic_error);
}

TEST_F(UnitPoolArenaTest, ClaimSplitsTheContainingFreeExtent) {
	// The whole arena is one free extent -- claim a sub-range out of its
	// middle and verify both flanking remainders are still free.
	arena_.claim(UnitPoolArena::UnitRange{3, 4});  // consumes [3,7)
	EXPECT_EQ(arena_.totalFreeUnits(), 6u);         // [0,3) + [7,10)
	EXPECT_EQ(arena_.largestFreeExtent(), 3u);

	auto extents = arena_.freeExtentsByAddress();
	ASSERT_EQ(extents.size(), 2u);
	EXPECT_EQ(extents[0].start_unit, 0u);
	EXPECT_EQ(extents[0].unit_count, 3u);
	EXPECT_EQ(extents[1].start_unit, 7u);
	EXPECT_EQ(extents[1].unit_count, 3u);
}

TEST_F(UnitPoolArenaTest, FreeExtentsByAddressIsAddressOrdered) {
	// Five 2-unit pieces filling the whole 10-unit arena, every other one
	// freed -- p1 ([2,4)) and p3 ([6,8)) stay occupied as walls, so the three
	// freed pieces can't coalesce with each other or with any "never
	// allocated" tail (there isn't one -- the arena is exactly full).
	std::vector<UnitPoolArena::UnitRange> pieces;
	for (int i = 0; i < 5; ++i) {
		auto piece = arena_.allocateBestFit(2);
		ASSERT_TRUE(piece.has_value());
		pieces.push_back(*piece);
	}
	arena_.free(pieces[4]);  // freed out of address order on purpose
	arena_.free(pieces[0]);
	arena_.free(pieces[2]);

	auto extents = arena_.freeExtentsByAddress();
	ASSERT_EQ(extents.size(), 3u);
	EXPECT_EQ(extents[0].start_unit, 0u);
	EXPECT_EQ(extents[1].start_unit, 4u);
	EXPECT_EQ(extents[2].start_unit, 8u);
}

TEST_F(UnitPoolArenaTest, RelocateCopiesDataToTheNewLocation) {
	auto from = arena_.allocateBestFit(2);
	ASSERT_TRUE(from.has_value());

	std::vector<std::byte> pattern(2 * kUnitBytes, std::byte{0x5A});
	ASSERT_EQ(cudaMemcpy(arena_.pointerFor(*from), pattern.data(), pattern.size(), cudaMemcpyHostToDevice),
						cudaSuccess);

	UnitPoolArena::UnitRange to{5, 2};
	arena_.claim(to);
	arena_.relocate(*from, to, /*stream=*/nullptr);
	ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

	std::vector<std::byte> out(2 * kUnitBytes);
	ASSERT_EQ(cudaMemcpy(out.data(), arena_.pointerFor(to), out.size(), cudaMemcpyDeviceToHost), cudaSuccess);
	EXPECT_EQ(out, pattern);
}

}  // namespace
