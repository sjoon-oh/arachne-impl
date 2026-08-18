#include "core/scheduling_policy.hpp"

#include <optional>
#include <variant>

namespace arachne {

namespace {

ExecutionMode ModeOf(const ScheduledOperation& op) {
	if (std::holds_alternative<TraverseTask>(op)) return std::get<TraverseTask>(op).request.mode;
	return std::get<ModifyTask>(op).request.mode;
}

// Only ever called with a ScheduledOperation already known to hold a
// ModifyTask (canAppendToBatch() only reaches this after kind_matches has
// confirmed `candidate` and, transitively via the batch's own
// kind-homogeneity invariant, current_batch.front() both hold ModifyTask).
ModifyOp OpOf(const ScheduledOperation& op) { return std::get<ModifyTask>(op).request.op; }

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

	if (current_batch.empty()) return true;

	// Mode-homogeneity invariant -- see SchedulingPolicy's class doc comment.
	if (ModeOf(candidate) != ModeOf(current_batch.front())) return false;

	// Op-homogeneity invariant (Modify batches only) -- see SchedulingPolicy's
	// class doc comment.
	if (batch_kind == ScheduledKind::Modify && OpOf(candidate) != OpOf(current_batch.front())) {
		return false;
	}

	return true;
}

}  // namespace arachne

