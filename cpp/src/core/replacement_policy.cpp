#include "core/replacement_policy.hpp"

namespace arachne {

void FifoReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.push_back(std::move(candidate));
}

void FifoReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);

	// Might still be sitting unselected (never reached
	// selectNextPromotionCandidate() yet) -- e.g. requestPromotion() followed
	// quickly by a delete/verification-mismatch before the Coordinator got to
	// it. Drop it here too, not just from promoted_order_/promoted_tracked_
	// below, or a stale candidate would still get offered later.
	pending_candidates_.erase(std::remove_if(pending_candidates_.begin(), pending_candidates_.end(),
																						[anchor_id](const PromotionCandidate& candidate) {
																							return candidate.anchor_id == anchor_id;
																						}),
														 pending_candidates_.end());

	if (promoted_tracked_.erase(anchor_id) == 0) return;
	auto it = std::find(promoted_order_.begin(), promoted_order_.end(), anchor_id);
	if (it != promoted_order_.end()) promoted_order_.erase(it);
}

void FifoReplacementPolicy::onAnchorTouched(VectorId) {}

bool FifoReplacementPolicy::onRelocationTrigger() {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

bool FifoReplacementPolicy::hasPendingCandidates() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

std::optional<PromotionCandidate> FifoReplacementPolicy::selectNextPromotionCandidate() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (pending_candidates_.empty()) return std::nullopt;

	PromotionCandidate candidate = std::move(pending_candidates_.front());
	pending_candidates_.pop_front();

	// Record admission order for eviction purposes now, at selection time --
	// see the interface doc comment for why there is no separate grant-time
	// confirmation.
	if (promoted_tracked_.insert(candidate.anchor_id).second) {
		promoted_order_.push_back(candidate.anchor_id);
	}
	return candidate;
}

std::optional<VectorId> FifoReplacementPolicy::selectNextEvictionCandidate(VectorId excluded) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (VectorId candidate : promoted_order_) {
		if (candidate != excluded) return candidate;
	}
	return std::nullopt;
}

}  // namespace arachne
