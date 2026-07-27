#include "core/anchor_manager.hpp"

#include <gtest/gtest.h>

namespace {

using arachne::AnchorManager;
using arachne::LeaseHandle;
using arachne::Stitch;

TEST(AnchorManagerTest, StitchesOfIsEmptyForUnknownAnchor) {
	AnchorManager manager;
	EXPECT_TRUE(manager.stitchesOf(1).empty());
}

TEST(AnchorManagerTest, AddStitchIsVisibleViaStitchesOf) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/42, LeaseHandle{/*region=*/42, /*epoch=*/7});

	std::vector<Stitch> stitches = manager.stitchesOf(1);

	ASSERT_EQ(stitches.size(), 1u);
	EXPECT_EQ(stitches[0].region, 42u);
	EXPECT_EQ(stitches[0].lease.epoch, 7u);
}

TEST(AnchorManagerTest, AddStitchForSameRegionIsANoop) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/42, LeaseHandle{/*region=*/42, /*epoch=*/7});
	manager.addStitch(1, /*region=*/42, LeaseHandle{/*region=*/42, /*epoch=*/8});

	EXPECT_EQ(manager.stitchesOf(1).size(), 1u);
	EXPECT_EQ(manager.stitchesOf(1)[0].lease.epoch, 7u);  // first write wins, no duplicate
}

TEST(AnchorManagerTest, DistinctRegionsAccumulateSeparateStitches) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/10, LeaseHandle{10, 1});
	manager.addStitch(1, /*region=*/20, LeaseHandle{20, 2});

	EXPECT_EQ(manager.stitchesOf(1).size(), 2u);
}

TEST(AnchorManagerTest, RemoveStitchReturnsItAndDropsIt) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/42, LeaseHandle{42, 7});

	Stitch removed = manager.removeStitch(1, 42);

	EXPECT_TRUE(removed.lease.valid());
	EXPECT_EQ(removed.region, 42u);
	EXPECT_EQ(removed.lease.epoch, 7u);
	EXPECT_TRUE(manager.stitchesOf(1).empty());
}

TEST(AnchorManagerTest, RemoveStitchOfUnknownRegionReturnsInvalidStitch) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/42, LeaseHandle{42, 7});

	Stitch removed = manager.removeStitch(1, /*region=*/99);

	EXPECT_FALSE(removed.lease.valid());
	EXPECT_EQ(manager.stitchesOf(1).size(), 1u);  // untouched
}

TEST(AnchorManagerTest, ForgetRemovesAndReturnsEverySitch) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/10, LeaseHandle{10, 1});
	manager.addStitch(1, /*region=*/20, LeaseHandle{20, 2});

	std::vector<Stitch> forgotten = manager.forget(1);

	EXPECT_EQ(forgotten.size(), 2u);
	EXPECT_TRUE(manager.stitchesOf(1).empty());
}

TEST(AnchorManagerTest, ForgetOfUnknownAnchorIsANoop) {
	AnchorManager manager;
	EXPECT_TRUE(manager.forget(999).empty());
}

TEST(AnchorManagerTest, DifferentAnchorsAreIndependent) {
	AnchorManager manager;
	manager.addStitch(1, /*region=*/10, LeaseHandle{10, 1});
	manager.addStitch(2, /*region=*/10, LeaseHandle{10, 2});

	manager.forget(1);

	EXPECT_TRUE(manager.stitchesOf(1).empty());
	EXPECT_EQ(manager.stitchesOf(2).size(), 1u);
}

}  // namespace
