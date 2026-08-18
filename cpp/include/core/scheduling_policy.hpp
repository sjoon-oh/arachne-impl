#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <optional>
#include <variant>
#include <vector>

#include "adapter/index_adapter.hpp"

namespace arachne {

enum class ScheduledKind { Traverse, Modify };

struct TraverseTask {
	std::uint64_t id = 0;
	std::chrono::steady_clock::time_point enqueued_at;
	TraverseRequest request;
	std::promise<TraverseResult> promise;
	// Invoked on the execution worker thread, right before the promise is
	// fulfilled -- see OpScheduler::schedule(TraverseRequest, ...)'s doc
	// comment. OpScheduler never interprets this itself, same as start()'s
	// on_worker_start callback.
	std::function<void(const TraverseResult&)> on_complete;
};

struct ModifyTask {
	std::uint64_t id = 0;
	std::chrono::steady_clock::time_point enqueued_at;
	ModifyRequest request;
	std::promise<ModifyResult> promise;
};

using ScheduledOperation = std::variant<TraverseTask, ModifyTask>;
using ScheduledOperationQueue = std::deque<ScheduledOperation>;
using ScheduledOperationBatch = std::vector<ScheduledOperation>;

/// Pluggable dispatch/reorder policy consulted by OpScheduler's planner
/// thread (see op_scheduler.hpp) each time it needs to build a new batch out
/// of the pending ScheduledOperationQueue: chooseBatchKind() picks Traverse
/// vs. Modify, then selectCandidateIndex()/canAppendToBatch() are polled
/// repeatedly to grow the batch one op at a time until it's full or no
/// eligible candidate remains.
///
/// Every implementation must enforce two invariants regardless of its own
/// ordering strategy:
///  - A batch must be mode-homogeneous, not just kind-homogeneous.
///    OpScheduler dispatches a whole batch to exactly one of IAdapter's
///    Host/Device entry points (adapter/index_adapter.hpp), so a candidate
///    whose ExecutionMode (TraverseRequest::mode/ModifyRequest::mode)
///    doesn't match the batch-in-progress's must be rejected by
///    canAppendToBatch() -- there would be no single call correct for a
///    mixed batch otherwise.
///  - A Modify batch must also be op-homogeneous (ModifyRequest::op):
///    Insert and Delete are different operations against the index (Insert
///    is itself Traversal-then-Modification, Delete is Modification-only --
///    see IAdapter's own doc comment) with different request shapes and
///    different routing (e.g. Controller::routeRemove() never considers
///    GpuOnly, routeInsert() does), so a candidate whose op doesn't match
///    the batch-in-progress's must be rejected the same way a mode mismatch
///    is -- this lets an adapter's modifyHost()/modifyDevice() assume every
///    request in one call shares the same op, without weakening the
///    existing IAdapter contract (a single batch call per Host/Device
///    invocation) to do it.
class SchedulingPolicy {
 public:
	virtual ~SchedulingPolicy() = default;

	/// Chooses what kind of batch to emit next from the queue.
	virtual ScheduledKind chooseBatchKind(const ScheduledOperationQueue& queue) const = 0;

	/// Chooses which index in queue to append to the current batch. Return
	/// `nullopt` when no eligible candidate exists.
	virtual std::optional<std::size_t> selectCandidateIndex(
			const ScheduledOperationQueue& queue, ScheduledKind batch_kind,
			const ScheduledOperationBatch& current_batch) const = 0;

	/// Additional per-candidate validation before appending -- see class doc
	/// comment for the mode-homogeneity invariant every implementation must
	/// enforce here at minimum.
	virtual bool canAppendToBatch(ScheduledKind batch_kind, const ScheduledOperation& candidate,
																const ScheduledOperationBatch& current_batch) const = 0;
};

/// Default policy: preserves enqueue order by taking only the first eligible
/// kind from the queue.
class FifoSchedulingPolicy final : public SchedulingPolicy {
 public:
	ScheduledKind chooseBatchKind(const ScheduledOperationQueue& queue) const override;

	std::optional<std::size_t> selectCandidateIndex(const ScheduledOperationQueue& queue,
																									ScheduledKind batch_kind,
																									const ScheduledOperationBatch& current_batch) const override;

	bool canAppendToBatch(ScheduledKind batch_kind, const ScheduledOperation& candidate,
												const ScheduledOperationBatch& current_batch) const override;
};

}  // namespace arachne

