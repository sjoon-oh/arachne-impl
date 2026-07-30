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

void LruReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.push_back(std::move(candidate));
}

void LruReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);

	// Same reasoning as FifoReplacementPolicy::onAnchorEvicted(): a candidate
	// may still be sitting unselected.
	pending_candidates_.erase(std::remove_if(pending_candidates_.begin(), pending_candidates_.end(),
																						[anchor_id](const PromotionCandidate& candidate) {
																							return candidate.anchor_id == anchor_id;
																						}),
														 pending_candidates_.end());

	auto it = lru_position_.find(anchor_id);
	if (it == lru_position_.end()) return;
	lru_order_.erase(it->second);
	lru_position_.erase(it);
}

void LruReplacementPolicy::onAnchorTouched(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = lru_position_.find(anchor_id);
	if (it == lru_position_.end()) return;  // not currently resident -- nothing to reorder

	// O(1) move to the most-recently-used (back) end -- splice() relinks the
	// existing node in place rather than erasing/reinserting.
	lru_order_.splice(lru_order_.end(), lru_order_, it->second);
}

bool LruReplacementPolicy::onRelocationTrigger() {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

bool LruReplacementPolicy::hasPendingCandidates() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

std::optional<PromotionCandidate> LruReplacementPolicy::selectNextPromotionCandidate() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (pending_candidates_.empty()) return std::nullopt;

	PromotionCandidate candidate = std::move(pending_candidates_.front());
	pending_candidates_.pop_front();

	// Becoming resident counts as a "use" -- insert at the most-recently-used
	// end now, at selection time (mirrors FifoReplacementPolicy's own
	// admission-order recording here -- see the interface doc comment for why
	// there is no separate grant-time confirmation). A second selection for
	// an already-tracked Anchor (e.g. a second Region) leaves its recency
	// alone; onAnchorTouched() is the intended signal for that.
	if (lru_position_.find(candidate.anchor_id) == lru_position_.end()) {
		lru_order_.push_back(candidate.anchor_id);
		lru_position_.emplace(candidate.anchor_id, std::prev(lru_order_.end()));
	}
	return candidate;
}

std::optional<VectorId> LruReplacementPolicy::selectNextEvictionCandidate(VectorId excluded) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (VectorId candidate : lru_order_) {
		if (candidate != excluded) return candidate;
	}
	return std::nullopt;
}

// ---------------------------------------------------------------------------
// LfuReplacementPolicy
// ---------------------------------------------------------------------------

void LfuReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.push_back(std::move(candidate));
}

void LfuReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.erase(std::remove_if(pending_candidates_.begin(), pending_candidates_.end(),
																						[anchor_id](const PromotionCandidate& candidate) {
																							return candidate.anchor_id == anchor_id;
																						}),
														 pending_candidates_.end());

	auto it = tracked_.find(anchor_id);
	if (it == tracked_.end()) return;
	auto bucket_it = freq_buckets_.find(it->second.freq);
	bucket_it->second.erase(it->second.pos);
	if (bucket_it->second.empty()) freq_buckets_.erase(bucket_it);
	tracked_.erase(it);
}

void LfuReplacementPolicy::onAnchorTouched(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = tracked_.find(anchor_id);
	if (it == tracked_.end()) return;  // not currently resident -- nothing to bump

	TrackedEntry& entry = it->second;
	auto old_bucket_it = freq_buckets_.find(entry.freq);
	old_bucket_it->second.erase(entry.pos);
	if (old_bucket_it->second.empty()) freq_buckets_.erase(old_bucket_it);

	++entry.freq;
	std::list<VectorId>& new_bucket = freq_buckets_[entry.freq];  // default-constructs an empty list if new
	new_bucket.push_back(anchor_id);
	entry.pos = std::prev(new_bucket.end());
}

bool LfuReplacementPolicy::onRelocationTrigger() {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

bool LfuReplacementPolicy::hasPendingCandidates() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

std::optional<PromotionCandidate> LfuReplacementPolicy::selectNextPromotionCandidate() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (pending_candidates_.empty()) return std::nullopt;

	PromotionCandidate candidate = std::move(pending_candidates_.front());
	pending_candidates_.pop_front();

	// Becoming resident starts an Anchor at frequency 1 -- an implicit first
	// "use", mirroring LRU/Clock's own treatment of grant-time (see the
	// interface doc comment for why there is no separate grant-time
	// confirmation). A second selection for an already-tracked Anchor (e.g. a
	// second Region) leaves its frequency alone; onAnchorTouched() is the
	// intended signal for that.
	if (tracked_.find(candidate.anchor_id) == tracked_.end()) {
		std::list<VectorId>& bucket = freq_buckets_[1];
		bucket.push_back(candidate.anchor_id);
		tracked_.emplace(candidate.anchor_id, TrackedEntry{1, std::prev(bucket.end())});
	}
	return candidate;
}

std::optional<VectorId> LfuReplacementPolicy::selectNextEvictionCandidate(VectorId excluded) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& [freq, bucket] : freq_buckets_) {  // ascending frequency order
		for (VectorId candidate : bucket) {
			if (candidate != excluded) return candidate;
		}
	}
	return std::nullopt;
}

// ---------------------------------------------------------------------------
// ClockReplacementPolicy
// ---------------------------------------------------------------------------

void ClockReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.push_back(std::move(candidate));
}

void ClockReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.erase(std::remove_if(pending_candidates_.begin(), pending_candidates_.end(),
																						[anchor_id](const PromotionCandidate& candidate) {
																							return candidate.anchor_id == anchor_id;
																						}),
														 pending_candidates_.end());

	auto it = position_.find(anchor_id);
	if (it == position_.end()) return;

	// O(1) removal: swap the removed slot with the last one, then pop_back --
	// see the class doc comment for why ring_ never needed to preserve
	// insertion order in the first place.
	std::size_t idx = it->second;
	std::size_t last = ring_.size() - 1;
	if (idx != last) {
		ring_[idx] = ring_[last];
		position_[ring_[idx].anchor_id] = idx;
	}
	ring_.pop_back();
	position_.erase(it);
	if (hand_ >= ring_.size()) hand_ = 0;
}

void ClockReplacementPolicy::onAnchorTouched(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = position_.find(anchor_id);
	if (it == position_.end()) return;  // not currently resident -- nothing to mark
	ring_[it->second].referenced = true;
}

bool ClockReplacementPolicy::onRelocationTrigger() {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

bool ClockReplacementPolicy::hasPendingCandidates() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

std::optional<PromotionCandidate> ClockReplacementPolicy::selectNextPromotionCandidate() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (pending_candidates_.empty()) return std::nullopt;

	PromotionCandidate candidate = std::move(pending_candidates_.front());
	pending_candidates_.pop_front();

	// Becoming resident starts an Anchor with its reference bit already set
	// (one free pass before the sweep can evict it) -- mirrors LRU/LFU's own
	// treatment of grant-time as an implicit first "use". A second selection
	// for an already-tracked Anchor leaves its bit alone.
	if (position_.find(candidate.anchor_id) == position_.end()) {
		position_.emplace(candidate.anchor_id, ring_.size());
		ring_.push_back(ClockEntry{candidate.anchor_id, /*referenced=*/true});
	}
	return candidate;
}

std::optional<VectorId> ClockReplacementPolicy::selectNextEvictionCandidate(VectorId excluded) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (ring_.empty()) return std::nullopt;

	// Bounded by two full sweeps: every entry can only be spared once (its
	// bit cleared on the first pass) before the second pass finds it
	// unreferenced -- see the class doc comment.
	std::size_t limit = 2 * ring_.size();
	for (std::size_t steps = 0; steps < limit; ++steps) {
		if (hand_ >= ring_.size()) hand_ = 0;
		ClockEntry& entry = ring_[hand_];

		if (entry.anchor_id == excluded) {
			hand_ = (hand_ + 1) % ring_.size();
			continue;
		}
		if (entry.referenced) {
			entry.referenced = false;  // second chance: spared once, bit cleared
			hand_ = (hand_ + 1) % ring_.size();
			continue;
		}
		return entry.anchor_id;  // unreferenced and not excluded -- victim found; hand_ stays here
	}
	return std::nullopt;  // everything still standing is excluded
}

// ---------------------------------------------------------------------------
// TwoQReplacementPolicy
// ---------------------------------------------------------------------------

TwoQReplacementPolicy::TwoQReplacementPolicy(std::size_t ghost_capacity) : ghost_capacity_(ghost_capacity) {}

void TwoQReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.push_back(std::move(candidate));
}

void TwoQReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.erase(std::remove_if(pending_candidates_.begin(), pending_candidates_.end(),
																						[anchor_id](const PromotionCandidate& candidate) {
																							return candidate.anchor_id == anchor_id;
																						}),
														 pending_candidates_.end());

	if (auto it = a1in_position_.find(anchor_id); it != a1in_position_.end()) {
		a1in_.erase(it->second);
		a1in_position_.erase(it);

		// Remember the departure in the bounded ghost queue -- see the class
		// doc comment for why only a1in_ departures are remembered.
		a1out_.push_back(anchor_id);
		a1out_position_.emplace(anchor_id, std::prev(a1out_.end()));
		if (a1out_.size() > ghost_capacity_) {
			VectorId oldest = a1out_.front();
			a1out_.pop_front();
			a1out_position_.erase(oldest);
		}
		return;
	}

	if (auto it = am_position_.find(anchor_id); it != am_position_.end()) {
		am_.erase(it->second);
		am_position_.erase(it);
		return;
	}

	if (auto it = a1out_position_.find(anchor_id); it != a1out_position_.end()) {
		a1out_.erase(it->second);
		a1out_position_.erase(it);
	}
}

void TwoQReplacementPolicy::onAnchorTouched(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);

	if (auto it = a1in_position_.find(anchor_id); it != a1in_position_.end()) {
		// Touched a second time while still a first-timer -- proven itself,
		// promote to the protected am_ queue.
		a1in_.erase(it->second);
		a1in_position_.erase(it);
		am_.push_back(anchor_id);
		am_position_.emplace(anchor_id, std::prev(am_.end()));
		return;
	}

	if (auto it = am_position_.find(anchor_id); it != am_position_.end()) {
		am_.splice(am_.end(), am_, it->second);  // ordinary LRU touch within am_
		return;
	}
	// Not currently resident (or sitting only in the ghost queue) -- nothing
	// to do, same as every other policy's untracked-touch no-op.
}

bool TwoQReplacementPolicy::onRelocationTrigger() {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

bool TwoQReplacementPolicy::hasPendingCandidates() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

std::optional<PromotionCandidate> TwoQReplacementPolicy::selectNextPromotionCandidate() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (pending_candidates_.empty()) return std::nullopt;

	PromotionCandidate candidate = std::move(pending_candidates_.front());
	pending_candidates_.pop_front();

	VectorId anchor_id = candidate.anchor_id;
	if (a1in_position_.find(anchor_id) != a1in_position_.end() ||
			am_position_.find(anchor_id) != am_position_.end()) {
		return candidate;  // already resident (e.g. a second Region) -- leave placement alone
	}

	if (auto it = a1out_position_.find(anchor_id); it != a1out_position_.end()) {
		// A returning visitor -- already proved itself before being evicted
		// from a1in_ once, so skip straight to the protected am_ queue.
		a1out_.erase(it->second);
		a1out_position_.erase(it);
		am_.push_back(anchor_id);
		am_position_.emplace(anchor_id, std::prev(am_.end()));
	} else {
		a1in_.push_back(anchor_id);
		a1in_position_.emplace(anchor_id, std::prev(a1in_.end()));
	}
	return candidate;
}

std::optional<VectorId> TwoQReplacementPolicy::selectNextEvictionCandidate(VectorId excluded) {
	std::lock_guard<std::mutex> lock(mutex_);
	// a1in_ first (cheapest, safest sacrifice), am_ only once a1in_ is
	// exhausted -- see the class doc comment for why this ordering alone is
	// what protects am_ from scan pollution.
	for (VectorId candidate : a1in_) {
		if (candidate != excluded) return candidate;
	}
	for (VectorId candidate : am_) {
		if (candidate != excluded) return candidate;
	}
	return std::nullopt;
}

}  // namespace arachne
