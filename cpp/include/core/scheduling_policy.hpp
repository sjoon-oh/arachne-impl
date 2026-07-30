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
	// Invoked on the execution worker thread, with the freshly computed
	// result, right before the promise is fulfilled -- see
	// OpScheduler::schedule(TraverseRequest, ...)'s own doc comment for why:
	// this lets a caller (Controller) do result-dependent bookkeeping on the
	// bounded set of worker threads instead of on whichever (unboundedly
	// many, concurrent) thread happens to call future.get(). OpScheduler
	// itself never interprets this -- same shape as start()'s
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

/// Pluggable dispatch/reorder policy for scheduling.
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

	/// Additional per-candidate validation before appending. Implementations
	/// must, at minimum, reject a candidate whose ExecutionMode
	/// (TraverseRequest::mode/ModifyRequest::mode) doesn't match
	/// `current_batch`'s (once non-empty) -- OpScheduler dispatches a whole
	/// batch to exactly one of IAdapter's Host/Device entry points (see
	/// adapter/index_adapter.hpp), so a batch must be mode-homogeneous, not
	/// just kind-homogeneous, or there'd be no single call that's correct
	/// for the whole thing.
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

