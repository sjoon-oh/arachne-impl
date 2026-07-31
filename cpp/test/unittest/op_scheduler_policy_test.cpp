#include "core/scheduling_policy.hpp"

#include <cstdint>
#include <gtest/gtest.h>

namespace {

using arachne::FifoSchedulingPolicy;
using arachne::ScheduledKind;
using arachne::ScheduledOperation;
using arachne::ScheduledOperationBatch;
using arachne::ScheduledOperationQueue;

namespace {

	arachne::TraverseTask makeTraverse(std::uint64_t id) {
	arachne::TraverseTask task;
		task.id = id;
		return task;
	}

	arachne::ModifyTask makeModify(std::uint64_t id) {
	arachne::ModifyTask task;
		task.id = id;
		return task;
	}

}  // namespace

// Validates FifoSchedulingPolicy's queue-inspection helpers in isolation
// (chooseBatchKind/selectCandidateIndex/canAppendToBatch) against a plain
// ScheduledOperationQueue/Batch, without a running OpScheduler -- i.e. the
// policy decisions themselves, not their effect on live scheduling.

TEST(FifoSchedulingPolicyTest, ChooseBatchKindUsesFrontTypeOrDefaultsToTraverse) {
	FifoSchedulingPolicy policy;
	ScheduledOperationQueue queue;

	EXPECT_EQ(policy.chooseBatchKind(queue), ScheduledKind::Traverse);

	queue.push_back(makeTraverse(1));
	EXPECT_EQ(policy.chooseBatchKind(queue), ScheduledKind::Traverse);

	queue.clear();
	queue.push_back(makeModify(2));
	EXPECT_EQ(policy.chooseBatchKind(queue), ScheduledKind::Modify);
}

TEST(FifoSchedulingPolicyTest, SelectCandidateIndexFindsFirstMatchingKind) {
	FifoSchedulingPolicy policy;
	ScheduledOperationQueue queue;
	ScheduledOperationBatch batch;
	queue.push_back(makeTraverse(1));
	queue.push_back(makeModify(2));
	queue.push_back(makeTraverse(3));
	queue.push_back(makeModify(4));

	auto traverse_index = policy.selectCandidateIndex(queue, ScheduledKind::Traverse, batch);
	ASSERT_TRUE(traverse_index.has_value());
	EXPECT_EQ(*traverse_index, 0u);

	auto modify_index = policy.selectCandidateIndex(queue, ScheduledKind::Modify, batch);
	ASSERT_TRUE(modify_index.has_value());
	EXPECT_EQ(*modify_index, 1u);
}

TEST(FifoSchedulingPolicyTest, CanAppendRejectsDifferentKinds) {
	FifoSchedulingPolicy policy;
	ScheduledOperation traverse = makeTraverse(1);
	ScheduledOperation modify = makeModify(2);
	ScheduledOperationBatch batch;

	EXPECT_TRUE(policy.canAppendToBatch(ScheduledKind::Traverse, traverse, batch));
	EXPECT_FALSE(policy.canAppendToBatch(ScheduledKind::Traverse, modify, batch));
	EXPECT_TRUE(policy.canAppendToBatch(ScheduledKind::Modify, modify, batch));
	EXPECT_FALSE(policy.canAppendToBatch(ScheduledKind::Modify, traverse, batch));
}

}  // namespace
