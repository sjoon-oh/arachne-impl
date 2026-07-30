#include "core/replacement_policy.hpp"

#include <gtest/gtest.h>

namespace {

using arachne::FifoReplacementPolicy;
using arachne::VectorId;

TEST(FifoReplacementPolicyTest, SelectEvictionCandidateIsNulloptWhenEmpty) {
	FifoReplacementPolicy policy;
	EXPECT_FALSE(policy.selectEvictionCandidate(/*excluded=*/0).has_value());
}

TEST(FifoReplacementPolicyTest, SelectEvictionCandidateReturnsOldestTracked) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);
	policy.onAnchorPromoted(2);
	policy.onAnchorPromoted(3);

	auto candidate = policy.selectEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, RepeatedPromotionDoesNotReorder) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);
	policy.onAnchorPromoted(2);
	policy.onAnchorPromoted(1);  // second dependency for the same Anchor: no reorder

	auto candidate = policy.selectEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, SelectEvictionCandidateSkipsExcluded) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);
	policy.onAnchorPromoted(2);

	auto candidate = policy.selectEvictionCandidate(/*excluded=*/1);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(FifoReplacementPolicyTest, SelectEvictionCandidateIsNulloptWhenOnlyExcludedIsTracked) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);

	EXPECT_FALSE(policy.selectEvictionCandidate(/*excluded=*/1).has_value());
}

TEST(FifoReplacementPolicyTest, AnchorEvictedStopsItFromBeingSelected) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);
	policy.onAnchorPromoted(2);

	policy.onAnchorEvicted(1);

	auto candidate = policy.selectEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(FifoReplacementPolicyTest, AnchorEvictedOfUntrackedAnchorIsANoop) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);

	policy.onAnchorEvicted(999);

	auto candidate = policy.selectEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(FifoReplacementPolicyTest, ReAddingAnEvictedAnchorGoesToTheBack) {
	FifoReplacementPolicy policy;
	policy.onAnchorPromoted(1);
	policy.onAnchorPromoted(2);

	policy.onAnchorEvicted(1);
	policy.onAnchorPromoted(1);  // re-promoted: should now be newer than 2

	auto candidate = policy.selectEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

}  // namespace
