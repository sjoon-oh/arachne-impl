#include "core/replacement_policy.hpp"

#include <algorithm>

namespace arachne {

void FifoReplacementPolicy::onAnchorPromoted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (tracked_.insert(anchor_id).second) {
		order_.push_back(anchor_id);
	}
}

void FifoReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (tracked_.erase(anchor_id) == 0) return;

	auto it = std::find(order_.begin(), order_.end(), anchor_id);
	if (it != order_.end()) order_.erase(it);
}

std::optional<VectorId> FifoReplacementPolicy::selectEvictionCandidate(VectorId excluded) const {
	std::lock_guard<std::mutex> lock(mutex_);
	for (VectorId candidate : order_) {
		if (candidate != excluded) return candidate;
	}
	return std::nullopt;
}

}  // namespace arachne
