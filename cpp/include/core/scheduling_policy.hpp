#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
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

	/// Additional per-candidate validation before appending.
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

