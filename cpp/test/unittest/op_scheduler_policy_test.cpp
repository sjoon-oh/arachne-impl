#include "core/scheduling_policy.hpp"

#include <cstdint>
#include <gtest/gtest.h>

namespace {

using arachne::ExecutionMode;
using arachne::FifoSchedulingPolicy;
using arachne::ModifyOp;
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

	arachne::ModifyTask makeModify(std::uint64_t id, ModifyOp op = ModifyOp::Insert,
																 ExecutionMode mode = ExecutionMode::Hybrid) {
	arachne::ModifyTask task;
		task.id = id;
		task.request.op = op;
		task.request.mode = mode;
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

// Op-homogeneity invariant (SchedulingPolicy's class doc comment): a Modify
// batch must not mix Insert and Delete, mirroring the pre-existing
// mode-homogeneity check above.
TEST(FifoSchedulingPolicyTest, CanAppendRejectsMixedOpsWithinModifyBatch) {
	FifoSchedulingPolicy policy;
	ScheduledOperationBatch batch;
	batch.push_back(makeModify(1, ModifyOp::Insert));

	ScheduledOperation another_insert = makeModify(2, ModifyOp::Insert);
	ScheduledOperation a_delete = makeModify(3, ModifyOp::Delete);

	EXPECT_TRUE(policy.canAppendToBatch(ScheduledKind::Modify, another_insert, batch))
			<< "same op (Insert) must still be allowed to join";
	EXPECT_FALSE(policy.canAppendToBatch(ScheduledKind::Modify, a_delete, batch))
			<< "Delete must not be allowed to join an Insert-only batch";
}

// Op-homogeneity is independent of, and checked alongside, mode-homogeneity
// -- same op but different mode must still be rejected.
TEST(FifoSchedulingPolicyTest, CanAppendStillEnforcesModeHomogeneityAlongsideOp) {
	FifoSchedulingPolicy policy;
	ScheduledOperationBatch batch;
	batch.push_back(makeModify(1, ModifyOp::Insert, ExecutionMode::Hybrid));

	ScheduledOperation same_op_different_mode = makeModify(2, ModifyOp::Insert, ExecutionMode::GpuOnly);
	EXPECT_FALSE(policy.canAppendToBatch(ScheduledKind::Modify, same_op_different_mode, batch));
}

// End-to-end through selectCandidateIndex(): with an Insert batch already
// started, a same-index Delete queued ahead of a later Insert must be
// skipped over (not picked, not silently accepted) -- the later Insert is
// what should be found, consistent with FifoSchedulingPolicy's own
// "preserves enqueue order by taking only the first *eligible*" contract.
TEST(FifoSchedulingPolicyTest, SelectCandidateIndexSkipsMismatchedOpToFindNextEligible) {
	FifoSchedulingPolicy policy;
	ScheduledOperationQueue queue;
	queue.push_back(makeModify(1, ModifyOp::Delete));
	queue.push_back(makeModify(2, ModifyOp::Insert));

	ScheduledOperationBatch batch;
	batch.push_back(makeModify(0, ModifyOp::Insert));

	auto index = policy.selectCandidateIndex(queue, ScheduledKind::Modify, batch);
	ASSERT_TRUE(index.has_value());
	EXPECT_EQ(*index, 1u) << "should skip the Delete at index 0 and find the Insert at index 1";
}

}  // namespace
