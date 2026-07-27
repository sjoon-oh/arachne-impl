#include "core/scheduling_policy.hpp"

#include <optional>

namespace arachne {

ScheduledKind FifoSchedulingPolicy::chooseBatchKind(const ScheduledOperationQueue& queue) const {
	if (queue.empty()) {
		return ScheduledKind::Traverse;
	}
	if (std::holds_alternative<TraverseTask>(queue.front())) {
		return ScheduledKind::Traverse;
	}
	return ScheduledKind::Modify;
}

std::optional<std::size_t> FifoSchedulingPolicy::selectCandidateIndex(
		const ScheduledOperationQueue& queue, ScheduledKind batch_kind,
		const ScheduledOperationBatch& current_batch) const {
	for (std::size_t i = 0; i < queue.size(); ++i) {
		if (!canAppendToBatch(batch_kind, queue[i], current_batch)) {
			continue;
		}
		return i;
	}
	return std::nullopt;
}

bool FifoSchedulingPolicy::canAppendToBatch(ScheduledKind batch_kind, const ScheduledOperation& candidate,
																			 const ScheduledOperationBatch& current_batch) const {
	(void)current_batch;
	if (batch_kind == ScheduledKind::Traverse) {
		return std::holds_alternative<TraverseTask>(candidate);
	}
	return std::holds_alternative<ModifyTask>(candidate);
}

}  // namespace arachne

