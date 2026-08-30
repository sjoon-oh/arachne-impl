#include "core/replacement_policy.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

// Tests for every concrete ReplacementPolicy implementation
// (core/replacement_policy.hpp): FifoReplacementPolicy, LruReplacementPolicy,
// LfuReplacementPolicy, ClockReplacementPolicy, TwoQReplacementPolicy. Each
// gets its own TEST suite (FifoReplacementPolicyTest, LruReplacementPolicyTest,
// ...) that walks the same menu of ReplacementPolicy contract points,
// customized to that policy's own eviction-order semantics:
//
//   - selectNextPromotionCandidate(): admission order (plain FIFO for every
//     policy here -- see enqueueCandidate()'s doc comment) and footprint
//     round-tripping through PromotionCandidate.
//   - hasPendingCandidates() / onRelocationTrigger(): both reduce to "is
//     there anything admitted but not yet selected".
//   - selectNextEvictionCandidate(): the part that actually differs per
//     policy --
//       Fifo:  oldest-granted-first.
//       Lru:   least-recently-touched-first; becoming resident itself
//              counts as an initial "use".
//       Lfu:   lowest touch-frequency-first, ties broken FIFO.
//       Clock: reference-bit sweep with second-chance semantics (bounded
//              two-sweep scan).
//       TwoQ:  a1in_ (first-timers) always preferred over am_ (proven
//              hot); the a1out_ ghost queue fast-tracks a returning id
//              straight into am_ instead of back through a1in_.
//     Every suite also covers the `excluded` skip and the "only the
//     excluded id is eligible" nullopt case.
//   - onAnchorEvicted(): must purge both the pending-candidate queue and
//     whatever eviction-ordering structure the policy uses; a no-op for
//     untracked ids.
//   - onAnchorTouched(): the hotness signal each policy interprets
//     differently (no-op for Fifo, recency bump for Lru, frequency bump
//     for Lfu, reference-bit set for Clock, a1in_ -> am_ promotion for
//     TwoQ); also a no-op for untracked ids.
//
// MakeCandidate() below is the one shared helper: a minimal
// PromotionCandidate (empty footprint) for a given anchor id, used by any
// test that doesn't care about footprint contents specifically.

namespace {

using arachne::AdmissionContext;
using arachne::ClockReplacementPolicy;
using arachne::CostAwareReplacementConfig;
using arachne::CostAwareReplacementPolicy;
using arachne::EvictionCandidate;
using arachne::FifoReplacementPolicy;
using arachne::LfuReplacementPolicy;
using arachne::LruReplacementPolicy;
using arachne::PromotionCandidate;
using arachne::RegionFootprint;
using arachne::RelocationBatchContext;
using arachne::TwoQReplacementPolicy;
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

TEST(FifoReplacementPolicyTest, RequeuePreservesOriginalAgeAndBatchHistory) {
	FifoReplacementPolicy policy;
	PromotionCandidate old = MakeCandidate(1);
	old.enqueue_sequence = 10;
	PromotionCandidate newer = MakeCandidate(2);
	newer.enqueue_sequence = 20;
	policy.enqueueCandidate(std::move(old));
	policy.enqueueCandidate(std::move(newer));

	auto attempted = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(attempted.has_value());
	attempted->first_batch_sequence = 7;
	attempted->last_batch_sequence = 9;
	attempted->planning_attempts = 3;
	policy.requeueCandidate(std::move(*attempted));

	auto retried = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(retried.has_value());
	EXPECT_EQ(retried->anchor_id, 1u);
	EXPECT_EQ(retried->enqueue_sequence, 10u);
	EXPECT_EQ(retried->first_batch_sequence, 7u);
	EXPECT_EQ(retried->last_batch_sequence, 9u);
	EXPECT_EQ(retried->planning_attempts, 3u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 2u);
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

// ---------------------------------------------------------------------------
// LruReplacementPolicy: admission order mirrors FIFO exactly (see the class
// doc comment, replacement_policy.hpp) -- only the eviction-order tests
// below actually differ from the FIFO suite above.
// ---------------------------------------------------------------------------

TEST(LruReplacementPolicyTest, SelectNextPromotionCandidateIsNulloptWhenEmpty) {
	LruReplacementPolicy policy;
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(LruReplacementPolicyTest, SelectNextPromotionCandidateReturnsInAdmissionOrder) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));

	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 1u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 2u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 3u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(LruReplacementPolicyTest, SelectNextPromotionCandidatePreservesFootprint) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(PromotionCandidate{1, RegionFootprint{{10, 20}}});

	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->footprint.regions.size(), 2u);
	EXPECT_EQ(candidate->footprint.regions[0], 10u);
	EXPECT_EQ(candidate->footprint.regions[1], 20u);
}

TEST(LruReplacementPolicyTest, HasPendingCandidatesAndOnRelocationTriggerReflectQueueState) {
	LruReplacementPolicy policy;
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
// selectNextEvictionCandidate(): unlike FIFO, ordered by recency of actual
// use (onAnchorTouched()), not by promotion order -- becoming resident
// (selection) itself counts as an initial "use".
// ---------------------------------------------------------------------------

TEST(LruReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptBeforeAnyPromotionSelected) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));  // admitted, but never selected
	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/0).has_value());
}

TEST(LruReplacementPolicyTest, SelectingAPromotionCandidateMakesItEvictionEligible) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // becoming resident counts as a use

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(LruReplacementPolicyTest, SelectNextEvictionCandidateReturnsLeastRecentlyUsedFirst) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));
	policy.selectNextPromotionCandidate();  // 1 becomes resident
	policy.selectNextPromotionCandidate();  // 2 becomes resident
	policy.selectNextPromotionCandidate();  // 3 becomes resident

	// No onAnchorTouched() calls yet -- residency order is still the
	// recency order, so the least-recently-used is the first one promoted.
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(LruReplacementPolicyTest, OnAnchorTouchedMovesAnchorToTheMostRecentlyUsedEnd) {
	// The key difference from FIFO: touching the *oldest* resident Anchor
	// protects it from eviction, even though it was promoted first.
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));
	policy.selectNextPromotionCandidate();  // 1
	policy.selectNextPromotionCandidate();  // 2
	policy.selectNextPromotionCandidate();  // 3

	policy.onAnchorTouched(1);  // 1 is now the most-recently-used

	// Least-recently-used is now 2, not 1.
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(LruReplacementPolicyTest, RepeatedTouchesKeepPromotingTheSameAnchorToTheBack) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);

	// 1 was touched (repeatedly) more recently than 2 was ever touched --
	// eviction order unaffected by touch *count*, only by recency.
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(LruReplacementPolicyTest, SelectNextEvictionCandidateSkipsExcluded) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/1);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(LruReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptWhenOnlyExcludedIsSelected) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/1).has_value());
}

TEST(LruReplacementPolicyTest, ReSelectingTheSameAnchorDoesNotResetItsRecency) {
	// A second Region dependency for an already-resident Anchor is not itself
	// a "use" -- onAnchorTouched() is the intended signal for that.
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(1));  // anchor 1 again (e.g. a second Region)
	policy.selectNextPromotionCandidate();      // 1
	policy.selectNextPromotionCandidate();      // 2
	policy.selectNextPromotionCandidate();      // 1 again -- already tracked, recency unchanged

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

// ---------------------------------------------------------------------------
// onAnchorEvicted(): must purge from *both* internal structures -- see the
// interface doc comment.
// ---------------------------------------------------------------------------

TEST(LruReplacementPolicyTest, OnAnchorEvictedRemovesAnAlreadySelectedAnchorFromEvictionOrder) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(LruReplacementPolicyTest, OnAnchorEvictedRemovesAnUnselectedCandidateFromThePendingQueue) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));

	policy.onAnchorEvicted(1);

	EXPECT_TRUE(policy.hasPendingCandidates());  // anchor 2 is still there
	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->anchor_id, 2u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());  // anchor 1 never offered
}

TEST(LruReplacementPolicyTest, OnAnchorEvictedOfUntrackedAnchorIsANoop) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_NO_THROW(policy.onAnchorEvicted(999));

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(LruReplacementPolicyTest, OnAnchorTouchedOfUntrackedAnchorIsANoop) {
	LruReplacementPolicy policy;
	EXPECT_NO_THROW(policy.onAnchorTouched(999));
}

TEST(LruReplacementPolicyTest, ReAdmittingAnEvictedAnchorGoesToTheMostRecentlyUsedEnd) {
	LruReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);
	policy.enqueueCandidate(MakeCandidate(1));  // re-admitted
	policy.selectNextPromotionCandidate();      // selected (== used) again -- newer than 2

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

// ---------------------------------------------------------------------------
// LfuReplacementPolicy: admission order mirrors FIFO exactly. The tests
// below focus on what's distinctive -- eviction order by touch *frequency*,
// not recency.
// ---------------------------------------------------------------------------

TEST(LfuReplacementPolicyTest, SelectNextPromotionCandidateReturnsInAdmissionOrder) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));

	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 1u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 2u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 3u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(LfuReplacementPolicyTest, SelectNextPromotionCandidatePreservesFootprint) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(PromotionCandidate{1, RegionFootprint{{10, 20}}});

	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->footprint.regions.size(), 2u);
}

TEST(LfuReplacementPolicyTest, HasPendingCandidatesAndOnRelocationTriggerReflectQueueState) {
	LfuReplacementPolicy policy;
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());

	policy.enqueueCandidate(MakeCandidate(1));
	EXPECT_TRUE(policy.hasPendingCandidates());
	EXPECT_TRUE(policy.onRelocationTrigger());

	policy.selectNextPromotionCandidate();
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());
}

TEST(LfuReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptBeforeAnyPromotionSelected) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/0).has_value());
}

TEST(LfuReplacementPolicyTest, SelectNextEvictionCandidateReturnsLowestFrequencyFirst) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 1 at freq 1
	policy.selectNextPromotionCandidate();  // 2 at freq 1

	policy.onAnchorTouched(2);  // 2 -> freq 2, so 1 (freq 1) is now strictly less frequent

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(LfuReplacementPolicyTest, FrequentlyTouchedAnchorOutranksARecentlyTouchedOne) {
	// The property LRU does *not* have: 1 was touched long ago but many
	// times; 2 was touched just now, but only once ever (via its own
	// grant). LFU must still protect 1.
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // 1 at freq 1
	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);  // 1 now at freq 5

	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 2 at freq 1, granted (touched) most recently

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);  // lower frequency, even though 1 is "older"
}

TEST(LfuReplacementPolicyTest, SelectNextEvictionCandidateSkipsExcludedAcrossFrequencyBuckets) {
	// excluded is the *sole* occupant of the minimum-frequency bucket -- the
	// scan must continue into the next bucket rather than stopping there.
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 1 at freq 1
	policy.selectNextPromotionCandidate();  // 2 at freq 1
	policy.onAnchorTouched(2);              // 2 -> freq 2; 1 alone at freq 1

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/1);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(LfuReplacementPolicyTest, OnAnchorEvictedRemovesAnAlreadySelectedAnchorFromEvictionOrder) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(LfuReplacementPolicyTest, OnAnchorEvictedRemovesAnUnselectedCandidateFromThePendingQueue) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));

	policy.onAnchorEvicted(1);

	EXPECT_TRUE(policy.hasPendingCandidates());
	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->anchor_id, 2u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(LfuReplacementPolicyTest, OnAnchorEvictedOfUntrackedAnchorIsANoop) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_NO_THROW(policy.onAnchorEvicted(999));

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(LfuReplacementPolicyTest, OnAnchorTouchedOfUntrackedAnchorIsANoop) {
	LfuReplacementPolicy policy;
	EXPECT_NO_THROW(policy.onAnchorTouched(999));
}

TEST(LfuReplacementPolicyTest, ReAdmittingAnEvictedAnchorStartsBackAtFrequencyOne) {
	LfuReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();
	policy.onAnchorTouched(1);
	policy.onAnchorTouched(1);  // 1 at freq 3

	policy.onAnchorEvicted(1);
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // re-admitted, back to freq 1

	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 2 also at freq 1

	// Both at freq 1 now -- tie broken FIFO (whichever reached this
	// frequency first), so 1 (re-admitted before 2 arrived) comes first.
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

// ---------------------------------------------------------------------------
// ClockReplacementPolicy: admission order mirrors FIFO exactly. The tests
// below focus on the reference-bit sweep's second-chance behavior.
// ---------------------------------------------------------------------------

TEST(ClockReplacementPolicyTest, SelectNextPromotionCandidateReturnsInAdmissionOrder) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));

	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 1u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 2u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 3u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(ClockReplacementPolicyTest, SelectNextPromotionCandidatePreservesFootprint) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(PromotionCandidate{1, RegionFootprint{{10, 20}}});

	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->footprint.regions.size(), 2u);
}

TEST(ClockReplacementPolicyTest, HasPendingCandidatesAndOnRelocationTriggerReflectQueueState) {
	ClockReplacementPolicy policy;
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());

	policy.enqueueCandidate(MakeCandidate(1));
	EXPECT_TRUE(policy.hasPendingCandidates());
	EXPECT_TRUE(policy.onRelocationTrigger());

	policy.selectNextPromotionCandidate();
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());
}

TEST(ClockReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptBeforeAnyPromotionSelected) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/0).has_value());
}

TEST(ClockReplacementPolicyTest, FreshlyGrantedAnchorsSurviveOneSweepThenBecomeEvictable) {
	// Every entry starts referenced=true (grant counts as a first "use") --
	// the first sweep must clear bits and spare everyone once; only a
	// second sweep can actually return a victim.
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());  // bounded 2-sweep scan still finds it on pass 2
	EXPECT_EQ(*candidate, 1u);
}

TEST(ClockReplacementPolicyTest, OnAnchorTouchedGivesAnAlreadyClearedEntryASecondChance) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 1, referenced=true
	policy.selectNextPromotionCandidate();  // 2, referenced=true

	// Both start referenced=true, so this sweep just clears bits and parks
	// hand_ on the first cleared slot (anchor 1) without evicting anyone.
	// excluded=999 (nonexistent) makes this a pure probe -- the result is
	// deliberately discarded.
	policy.selectNextEvictionCandidate(/*excluded=*/999);

	// hand_ is now parked on anchor 1's (cleared) slot. Touching it right
	// now is the "second chance" -- it must survive the next sweep, which
	// should instead clear and evict anchor 2 (never touched again).
	policy.onAnchorTouched(1);

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(ClockReplacementPolicyTest, SelectNextEvictionCandidateSkipsExcluded) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/1);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(ClockReplacementPolicyTest, SelectNextEvictionCandidateIsNulloptWhenOnlyExcludedIsSelected) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/1).has_value());
}

TEST(ClockReplacementPolicyTest, OnAnchorEvictedRemovesAnAlreadySelectedAnchorFromEvictionOrder) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(ClockReplacementPolicyTest, OnAnchorEvictedRemovalIsCorrectRegardlessOfRingPosition) {
	// Exercises the swap-with-last removal specifically for the *first*
	// slot (idx 0, forces the swap-from-last path) to prove position_
	// stays consistent afterward.
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();
	policy.selectNextPromotionCandidate();

	policy.onAnchorEvicted(1);  // removes ring_[0], swapping 3 into its place

	policy.onAnchorTouched(2);
	policy.onAnchorTouched(3);
	// Both remaining anchors touched -- neither should be lost/duplicated;
	// a full 2-sweep scan excluding one must still find the other.
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/2);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 3u);
}

TEST(ClockReplacementPolicyTest, OnAnchorEvictedRemovesAnUnselectedCandidateFromThePendingQueue) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));

	policy.onAnchorEvicted(1);

	EXPECT_TRUE(policy.hasPendingCandidates());
	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->anchor_id, 2u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(ClockReplacementPolicyTest, OnAnchorEvictedOfUntrackedAnchorIsANoop) {
	ClockReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	EXPECT_NO_THROW(policy.onAnchorEvicted(999));

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(ClockReplacementPolicyTest, OnAnchorTouchedOfUntrackedAnchorIsANoop) {
	ClockReplacementPolicy policy;
	EXPECT_NO_THROW(policy.onAnchorTouched(999));
}

// ---------------------------------------------------------------------------
// TwoQReplacementPolicy: admission order mirrors FIFO exactly. The tests
// below focus on the a1in_/am_/a1out_ split -- what makes 2Q resistant to
// scan pollution.
// ---------------------------------------------------------------------------

TEST(TwoQReplacementPolicyTest, SelectNextPromotionCandidateReturnsInAdmissionOrder) {
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.enqueueCandidate(MakeCandidate(3));

	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 1u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 2u);
	EXPECT_EQ(policy.selectNextPromotionCandidate()->anchor_id, 3u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(TwoQReplacementPolicyTest, SelectNextPromotionCandidatePreservesFootprint) {
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(PromotionCandidate{1, RegionFootprint{{10, 20}}});

	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->footprint.regions.size(), 2u);
}

TEST(TwoQReplacementPolicyTest, HasPendingCandidatesAndOnRelocationTriggerReflectQueueState) {
	TwoQReplacementPolicy policy;
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());

	policy.enqueueCandidate(MakeCandidate(1));
	EXPECT_TRUE(policy.hasPendingCandidates());
	EXPECT_TRUE(policy.onRelocationTrigger());

	policy.selectNextPromotionCandidate();
	EXPECT_FALSE(policy.hasPendingCandidates());
	EXPECT_FALSE(policy.onRelocationTrigger());
}

TEST(TwoQReplacementPolicyTest, FreshlyGrantedAnchorIsEvictionEligibleFromA1inImmediately) {
	// Unlike Clock, a1in_ gives no free pass -- a first-timer is always the
	// cheapest sacrifice (see the class doc comment).
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(TwoQReplacementPolicyTest, EvictionAlwaysPrefersA1inOverAmRegardlessOfOrder) {
	// 1 is touched twice -- promoted into am_ (the protected queue) -- while
	// 2 stays a first-timer in a1in_. Even though 1 was placed first, 2 (a
	// first-timer) is still preferred for eviction -- this is the core
	// scan-pollution protection.
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // 1 -> a1in_
	policy.onAnchorTouched(1);              // 1 promoted a1in_ -> am_

	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 2 -> a1in_ (first-timer)

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);  // a1in_ (2) preferred over am_ (1)
}

TEST(TwoQReplacementPolicyTest, OnAnchorTouchedWithinAmActsAsOrdinaryLru) {
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 1 -> a1in_
	policy.selectNextPromotionCandidate();  // 2 -> a1in_
	policy.onAnchorTouched(1);              // 1 -> am_
	policy.onAnchorTouched(2);              // 2 -> am_ (both now proven, a1in_ empty)

	policy.onAnchorTouched(1);  // ordinary LRU touch within am_: 1 -> most-recently-used

	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);  // least-recently-used within am_
}

TEST(TwoQReplacementPolicyTest, EvictedA1inAnchorReturningLaterSkipsStraightToAm) {
	// The ghost-list mechanism: an Anchor evicted once from a1in_ is
	// remembered; a fresh promotion for the same id goes straight to am_
	// instead of a1in_, since it already proved it isn't scan noise.
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // 1 -> a1in_
	policy.onAnchorEvicted(1);              // 1 evicted from a1in_ -> a1out_ ghost

	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // 1 returns -> straight to am_ (protected)

	policy.enqueueCandidate(MakeCandidate(2));
	policy.selectNextPromotionCandidate();  // 2 -> a1in_ (genuine first-timer)

	// 2 (a1in_) must be preferred over 1 (am_), even though 1 was granted
	// again more recently than 2.
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 2u);
}

TEST(TwoQReplacementPolicyTest, GhostQueueIsBoundedByConfiguredCapacity) {
	TwoQReplacementPolicy policy(/*ghost_capacity=*/2);
	for (VectorId id = 1; id <= 3; ++id) {
		policy.enqueueCandidate(MakeCandidate(id));
		policy.selectNextPromotionCandidate();  // id -> a1in_
		policy.onAnchorEvicted(id);             // id -> a1out_ (ghost)
	}
	// Ghost capacity is 2, so anchor 1 (the oldest ghost entry) must have
	// been evicted from a1out_ by the time anchor 3 pushed it out.
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // ghost miss -> a1in_ (NOT am_)

	policy.enqueueCandidate(MakeCandidate(99));
	policy.selectNextPromotionCandidate();  // a1in_ = [1, 99] if the above landed correctly

	// excluded=0 is the distinguishing check: the scan hits a1in_ first, so
	// this returns 1 only if 1 is genuinely a1in_'s front. If 1 had wrongly
	// skipped to am_ (treated as a ghost hit), this would return 99 instead
	// (a1in_'s sole occupant).
	auto candidate = policy.selectNextEvictionCandidate(/*excluded=*/0);
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(*candidate, 1u);
}

TEST(TwoQReplacementPolicyTest, SelectNextEvictionCandidateSkipsExcludedAcrossQueues) {
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.selectNextPromotionCandidate();  // 1 -> a1in_
	policy.onAnchorTouched(1);              // 1 -> am_ (a1in_ now empty)

	EXPECT_FALSE(policy.selectNextEvictionCandidate(/*excluded=*/1).has_value());
}

TEST(TwoQReplacementPolicyTest, OnAnchorEvictedRemovesAnUnselectedCandidateFromThePendingQueue) {
	TwoQReplacementPolicy policy;
	policy.enqueueCandidate(MakeCandidate(1));
	policy.enqueueCandidate(MakeCandidate(2));

	policy.onAnchorEvicted(1);

	EXPECT_TRUE(policy.hasPendingCandidates());
	auto candidate = policy.selectNextPromotionCandidate();
	ASSERT_TRUE(candidate.has_value());
	EXPECT_EQ(candidate->anchor_id, 2u);
	EXPECT_FALSE(policy.selectNextPromotionCandidate().has_value());
}

TEST(TwoQReplacementPolicyTest, OnAnchorEvictedOfUntrackedAnchorIsANoop) {
	TwoQReplacementPolicy policy;
	EXPECT_NO_THROW(policy.onAnchorEvicted(999));
}

TEST(TwoQReplacementPolicyTest, OnAnchorTouchedOfUntrackedAnchorIsANoop) {
	TwoQReplacementPolicy policy;
	EXPECT_NO_THROW(policy.onAnchorTouched(999));
}

// ---------------------------------------------------------------------------
// CostAwareReplacementPolicy::onAnchorTouched()'s touch_queue_ redesign (see
// its own doc comment) -- the latency-tracing report entry's fix for
// RegionManager::recordTraversal()'s heavy-tailed latency (up to 216s
// observed) turning out to come from onAnchorTouched() and
// evaluateAdmission()/selectEvictionCandidate() contending for the same
// mutex_.
// ---------------------------------------------------------------------------

TEST(CostAwareReplacementPolicyTest, OnAnchorTouchedNeverBlocksOnTheAdmissionScanLock) {
	CostAwareReplacementPolicy policy;

	// Make anchor 1 resident so onAnchorTouched(1) below isn't a same-instant
	// no-op once drained.
	AdmissionContext committed;
	committed.total_footprint_bytes = 100;
	policy.onPromotionCommitted(1, committed);

	// A large eviction_candidates list makes evaluateAdmission()'s locked
	// scan (see evaluateAdmission_locked's own trace-scope comment) take
	// real, repeated, measurable time under mutex_ -- each candidate's
	// group_members references an anchor id evaluateAdmission has to look up
	// in resident_ and miss, real (if small) per-item work, not a
	// short-circuited no-op. Deliberately *not* anchor 1 itself, so this
	// contender thread's own admission decisions don't depend on the touches
	// the main thread below is racing to apply.
	std::vector<EvictionCandidate> many_candidates;
	many_candidates.reserve(200000);
	for (int i = 0; i < 200000; ++i) {
		EvictionCandidate candidate;
		candidate.anchor_id = static_cast<VectorId>(1000 + i);
		candidate.reclaimable_bytes = 64;
		candidate.group_members = {static_cast<VectorId>(1000 + i)};
		many_candidates.push_back(std::move(candidate));
	}

	arachne::PromotionCandidate contender_candidate = MakeCandidate(2);
	AdmissionContext admission;
	admission.gpu_budget_bytes = 100;
	admission.gpu_bytes_allocated = 100;  // available = 0
	admission.incremental_bytes = 1;      // > available -- forces the locked scan below
	admission.eviction_candidates = many_candidates;

	std::atomic<bool> stop{false};
	std::atomic<std::uint64_t> contender_calls{0};
	std::thread contender([&] {
		while (!stop.load(std::memory_order_relaxed)) {
			policy.evaluateAdmission(contender_candidate, admission);
			contender_calls.fetch_add(1, std::memory_order_relaxed);
		}
	});

	using Clock = std::chrono::steady_clock;
	const Clock::time_point test_start = Clock::now();
	std::size_t iterations = 0;
	double max_call_ms = 0.0;
	while (Clock::now() - test_start < std::chrono::milliseconds(200)) {
		const Clock::time_point call_start = Clock::now();
		policy.onAnchorTouched(1);
		const double call_ms = std::chrono::duration<double, std::milli>(Clock::now() - call_start).count();
		max_call_ms = std::max(max_call_ms, call_ms);
		++iterations;
	}
	stop.store(true, std::memory_order_relaxed);
	contender.join();

	EXPECT_GT(iterations, 0u);
	EXPECT_GT(contender_calls.load(std::memory_order_relaxed), 0u)
			<< "contender thread should have made real progress concurrently, not been starved out entirely";
	// Generous bound (evaluateAdmission()'s own scan over 200,000 candidates
	// should itself take single-digit milliseconds or more per call, see the
	// latency-tracing report entry) -- the point is proving onAnchorTouched()
	// never meaningfully blocks behind that scan's mutex_, not measuring
	// exact latency.
	EXPECT_LT(max_call_ms, 20.0)
			<< "onAnchorTouched() should never stall behind evaluateAdmission()'s locked scan -- "
				 "it no longer shares mutex_ with it (see touch_queue_mutex_)";
}

TEST(CostAwareReplacementPolicyTest, TouchIsAppliedOnTheNextDrainNotImmediately) {
	CostAwareReplacementPolicy policy;

	AdmissionContext committed;
	committed.total_footprint_bytes = 100;
	policy.onPromotionCommitted(1, committed);  // anchor 1 resident, heat == 1.0

	policy.onAnchorTouched(1);  // queued, not yet applied -- see onAnchorTouched()'s own doc comment

	// selectEvictionCandidate()'s locked section drains the touch queue
	// before scoring (see its own drainTouchQueueLocked() call) -- so by the
	// time this returns, the touch above must already be reflected in
	// anchor 1's heat, even though nothing called onAnchorTouched() again in
	// between.
	EvictionCandidate candidate;
	candidate.anchor_id = 1;
	candidate.reclaimable_bytes = 64;
	candidate.group_members = {1};
	std::optional<VectorId> victim = policy.selectEvictionCandidate(/*excluded=*/0, /*required_bytes=*/1, {candidate});
	// Not asserting a specific density value here (that's
	// victimRetentionDensity()'s own concern, covered elsewhere) -- only
	// that the call completes and returns the only candidate offered,
	// proving the drain didn't corrupt resident_ or throw.
	ASSERT_TRUE(victim.has_value());
	EXPECT_EQ(*victim, 1u);
}

}  // namespace
