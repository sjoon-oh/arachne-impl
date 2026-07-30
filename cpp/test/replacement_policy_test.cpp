#include "core/replacement_policy.hpp"

#include <gtest/gtest.h>

namespace {

using arachne::FifoReplacementPolicy;
using arachne::PromotionCandidate;
using arachne::RegionFootprint;
using arachne::VectorId;

PromotionCandidate MakeCandidate(VectorId anchor_id) { return PromotionCandidate{anchor_id, RegionFootprint{}}; }

// ---------------------------------------------------------------------------
// selectNextPromotionCandidate(): admission order (the MPSC-queue-consumer
// side -- see ReplacementPolicy::enqueueCandidate()'s doc comment).
// ---------------------------------------------------------------------------

TEST(FifoReplacementPolicyTest, SelectNextPromotionCandidateIsNulloptWhenEmpty) {
	FifoReplacementPolicy policy;
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(FifoReplacementPolicyTest, SelectNextPromotionCandidateReturnsInAdmissionOrder) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));

	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 1u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 2u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 3u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(FifoReplacementPolicyTest, SelectNextPromotionCandidatePreservesFootprint) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(PromotionCandidate{1, RegionFootprint{{10, 20}}});

	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->footprint.regions.size(), 2u);
	EXPECT_EQ(candidate->footprint.regions[0], 10u);
	EXPECT_EQ(candidate->footprint.regions[1], 20u);
}

// ---------------------------------------------------------------------------
// hasPendingCandidates() / onRelocationTrigger(): both reduce to "is there
// anything admitted but not yet selected" for FIFO -- see the class doc
// comment for why FIFO never declines a trigger or holds candidates back.
// ---------------------------------------------------------------------------

TEST(FifoReplacementPolicyTest, HasPendingCandidatesAndOnRelocationTriggerReflectQueueState) {
	FifoReplacementPolicy policy;
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());

	policy.enqueueCandidate(MakeCandidate(1));
	EXPECT_TRUE(policy.hasPendingCandidates());
	EXPECT_TRUE(policy.onRelocationTrigger());

	policy.selectNextPromotionCandidate();
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());
}

// ---------------------------------------------------------------------------
// selectNextEvictionCandidate(): populated as a side effect of
// selectNextPromotionCandidate() (see that method's doc comment) -- there is
// no separate grant-time notification.
// ---------------------------------------------------------------------------

TEST(FifoReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptBeforeAnyPromotionSelected) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));  // admitted, but never selected
	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/0).has_value());
}

TEST(FifoReplacementPolicyTest, SelectingAPromotionCandidateMakesItEvictionEligible) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // records anchor 1 for eviction ordering

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, SelectNextEvictionCandidateReturnsOldestSelectedFirst) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, SelectNextEvictionCandidateSkipsExcluded) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/1);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(FifoReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptWhenOnlyExcludedIsSelected) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/1).has_value());
}

TEST(FifoReplacementPolicyTest, ReSelectingTheSameAnchorDoesNotReorderEvictionOrder) {
	// Mirrors the old RepeatedPromotionDoesNotReorder test: a second Region
	// dependency for an already-selected Anchor shouldn't change its eviction
	// standing.
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(1));  // anchor 1 again (e.g. a second Region)
	policy.selectNextPromotionCandidate();      // 1
	policy.selectNextPromotionCandidate();      // 2
	policy.selectNextPromotionCandidate();      // 1 again

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

// ---------------------------------------------------------------------------
// onAnchorEvicted(): must purge from *both* internal structures -- see the
// interface doc comment.
// ---------------------------------------------------------------------------

TEST(FifoReplacementPolicyTest, OnAnchorEvictedRemovesAnAlreadySelectedAnchorFromEvictionOrder) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(FifoReplacementPolicyTest, OnAnchorEvictedRemovesAnUnselectedCandidateFromThePendingQueue) {
	// The case the old design couldn't express: a candidate admitted but never
	// yet offered via selectNextPromotionCandidate() (e.g. requested, then
	// deleted, before the Coordinator got to it) must not still be offered
	// later.
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));

	policy.onAnchorEvicted(1);

	EXPECT_TRUE(policy.hasPendingCandidates());  // anchor 2 is still there
	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->anchor_id, 2u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());  // anchor 1 never offered
}

TEST(FifoReplacementPolicyTest, OnAnchorEvictedOfUntrackedAnchorIsANoop) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_NO_THROW(policy.onAnchorEvicted(999));

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, ReAdmittingAnEvictedAnchorGoesToTheBack) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);
	policy.enqueueCandidate(MakeCandidate(1));  // re-admitted
	policy.selectNextPromotionCandidate();      // selected again -- should now be newer than 2

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

// ---------------------------------------------------------------------------
// onAnchorTouched(): FIFO deliberately ignores usage frequency.
// ---------------------------------------------------------------------------

TEST(FifoReplacementPolicyTest, OnAnchorTouchedDoesNotAffectEvictionOrder) {
	FifoReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, OnAnchorTouchedOfUntrackedAnchorIsANoop) {
	FifoReplacementPolicy policy;
	EXPECT_NO_THROW(policy.onAnchorTouched(999));
}

}  // namespace
