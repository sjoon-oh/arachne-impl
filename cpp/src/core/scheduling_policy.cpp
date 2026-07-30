#include "core/scheduling_policy.hpp"

#include <optional>
#include <variant>

namespace arachne {

namespace {

ExecutionMode ModeOf(const ScheduledOperation& op) {
	if (std::holds_alternative<TraverseTask>(op)) return std::get<TraverseTask>(op).request.mode;
	return std::get<ModifyTask>(op).request.mode;
}

}  // namespace

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
	bool kind_matches = (batch_kind == ScheduledKind::Traverse)
													 ? std::holds_alternative<TraverseTask>(candidate)
													 : std::holds_alternative<ModifyTask>(candidate);
	if (!kind_matches) return false;

	// A batch is dispatched to exactly one IndexAdapter Host/Device entry
	// point, so every member must share the same ExecutionMode -- see
	// canAppendToBatch()'s doc comment.
	return current_batch.empty() || ModeOf(candidate) == ModeOf(current_batch.front());
}

}  // namespace arachne

