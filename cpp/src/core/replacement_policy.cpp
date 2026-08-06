#include "core/replacement_policy.hpp"

#include <cmath>
#include <limits>

namespace arachne {

namespace {

std::uint64_t CandidateOrder(const PromotionCandidate& candidate) {
	return candidate.enqueue_sequence == 0 ? std::numeric_limits<std::uint64_t>::max()
															 : candidate.enqueue_sequence;
}

void EnqueueByAge(std::deque<PromotionCandidate>& queue, PromotionCandidate candidate) {
	const std::uint64_t order = CandidateOrder(candidate);
	auto position = std::upper_bound(queue.begin(), queue.end(), order,
			[](std::uint64_t value, const PromotionCandidate& queued) {
				return value < CandidateOrder(queued);
			});
	queue.insert(position, std::move(candidate));
}

bool ContainsCandidate(const std::vector<EvictionCandidate>& candidates, VectorId anchor_id) {
	return std::any_of(candidates.begin(), candidates.end(), [anchor_id](const EvictionCandidate& candidate) {
		return candidate.anchor_id == anchor_id;
	});
}

}  // namespace

void FifoReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	EnqueueByAge(pending_candidates_, std::move(candidate));
}

void FifoReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);

	// May still be sitting unselected (requestPromotion() followed quickly by
	// a delete before the Coordinator got to it) -- drop it here too, not
	// just from promoted_order_/promoted_tracked_ below, or a stale candidate
	// would still get offered later.
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

std::optional<VectorId> FifoReplacementPolicy::selectEvictionCandidate(
		VectorId excluded, std::size_t, const std::vector<EvictionCandidate>& candidates) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (VectorId candidate : promoted_order_) {
		if (candidate != excluded && ContainsCandidate(candidates, candidate)) return candidate;
	}
	return std::nullopt;
}

void LruReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	EnqueueByAge(pending_candidates_, std::move(candidate));
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
	// end now, at selection time (see the interface doc comment for why there
	// is no separate grant-time confirmation). A second selection for an
	// already-tracked Anchor leaves its recency alone; onAnchorTouched() is
	// the intended signal for that.
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

std::optional<VectorId> LruReplacementPolicy::selectEvictionCandidate(
		VectorId excluded, std::size_t, const std::vector<EvictionCandidate>& candidates) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (VectorId candidate : lru_order_) {
		if (candidate != excluded && ContainsCandidate(candidates, candidate)) return candidate;
	}
	return std::nullopt;
}

// ---------------------------------------------------------------------------
// LfuReplacementPolicy
// ---------------------------------------------------------------------------

void LfuReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	EnqueueByAge(pending_candidates_, std::move(candidate));
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
	// "use" (see the interface doc comment for why there is no separate
	// grant-time confirmation). A second selection for an already-tracked
	// Anchor leaves its frequency alone; onAnchorTouched() is the intended
	// signal for that.
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

std::optional<VectorId> LfuReplacementPolicy::selectEvictionCandidate(
		VectorId excluded, std::size_t, const std::vector<EvictionCandidate>& candidates) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& [freq, bucket] : freq_buckets_) {
		for (VectorId candidate : bucket) {
			if (candidate != excluded && ContainsCandidate(candidates, candidate)) return candidate;
		}
	}
	return std::nullopt;
}

// ---------------------------------------------------------------------------
// ClockReplacementPolicy
// ---------------------------------------------------------------------------

void ClockReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	EnqueueByAge(pending_candidates_, std::move(candidate));
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

std::optional<VectorId> ClockReplacementPolicy::selectEvictionCandidate(
		VectorId excluded, std::size_t, const std::vector<EvictionCandidate>& candidates) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (ring_.empty()) return std::nullopt;

	const std::size_t max_steps = ring_.size() * 2;
	for (std::size_t step = 0; step < max_steps; ++step) {
		ClockEntry& entry = ring_[hand_];
		hand_ = (hand_ + 1) % ring_.size();
		if (entry.anchor_id == excluded || !ContainsCandidate(candidates, entry.anchor_id)) continue;
		if (entry.referenced) {
			entry.referenced = false;
			continue;
		}
		return entry.anchor_id;
	}
	return std::nullopt;
}

// ---------------------------------------------------------------------------
// TwoQReplacementPolicy
// ---------------------------------------------------------------------------

TwoQReplacementPolicy::TwoQReplacementPolicy(std::size_t ghost_capacity) : ghost_capacity_(ghost_capacity) {}

void TwoQReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	EnqueueByAge(pending_candidates_, std::move(candidate));
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

std::optional<VectorId> TwoQReplacementPolicy::selectEvictionCandidate(
		VectorId excluded, std::size_t, const std::vector<EvictionCandidate>& candidates) {
	std::lock_guard<std::mutex> lock(mutex_);
	for (VectorId candidate : a1in_) {
		if (candidate != excluded && ContainsCandidate(candidates, candidate)) return candidate;
	}
	for (VectorId candidate : am_) {
		if (candidate != excluded && ContainsCandidate(candidates, candidate)) return candidate;
	}
	return std::nullopt;
}

CostAwareReplacementPolicy::CostAwareReplacementPolicy(CostAwareReplacementConfig config)
		: config_(std::move(config)) {
	if (config_.admission_hysteresis < 0.0) config_.admission_hysteresis = 0.0;
	if (config_.potential_writeback_weight < 0.0) config_.potential_writeback_weight = 0.0;
}

void CostAwareReplacementPolicy::enqueueCandidate(PromotionCandidate candidate) {
	std::lock_guard<std::mutex> lock(mutex_);
	candidate.observations = std::max<std::uint64_t>(1, candidate.observations);
	for (PromotionCandidate& pending : pending_candidates_) {
		if (pending.anchor_id != candidate.anchor_id || pending.epoch != candidate.epoch) continue;
		pending.observations += candidate.observations;
		for (RegionId region : candidate.footprint.regions) {
			if (std::find(pending.footprint.regions.begin(), pending.footprint.regions.end(), region) ==
					pending.footprint.regions.end()) {
				pending.footprint.regions.push_back(region);
			}
		}
		if (!candidate.vector_bytes.empty()) {
			pending.vector_bytes = std::move(candidate.vector_bytes);
			pending.vector_dim = candidate.vector_dim;
			pending.vector_dtype = candidate.vector_dtype;
		}
		if (pending.enqueue_sequence == 0 ||
				(candidate.enqueue_sequence != 0 && candidate.enqueue_sequence < pending.enqueue_sequence)) {
			pending.enqueue_sequence = candidate.enqueue_sequence;
			pending.enqueued_at = candidate.enqueued_at;
		}
		if (pending.first_batch_sequence == 0 ||
				(candidate.first_batch_sequence != 0 &&
				 candidate.first_batch_sequence < pending.first_batch_sequence)) {
			pending.first_batch_sequence = candidate.first_batch_sequence;
		}
		pending.last_batch_sequence = std::max(pending.last_batch_sequence, candidate.last_batch_sequence);
		pending.planning_attempts = std::max(pending.planning_attempts, candidate.planning_attempts);
		return;
	}
	EnqueueByAge(pending_candidates_, std::move(candidate));
}

void CostAwareReplacementPolicy::onAnchorEvicted(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	pending_candidates_.erase(std::remove_if(pending_candidates_.begin(), pending_candidates_.end(),
																				[anchor_id](const PromotionCandidate& candidate) {
																					return candidate.anchor_id == anchor_id;
																				}),
										 pending_candidates_.end());
	resident_.erase(anchor_id);
}

double CostAwareReplacementPolicy::decayedHeat(const ResidentEntry& entry, Clock::time_point now) const {
	double half_life = std::chrono::duration<double>(config_.heat_half_life).count();
	if (half_life <= 0.0) return entry.heat;
	double elapsed = std::chrono::duration<double>(now - entry.last_update).count();
	return entry.heat * std::exp2(-elapsed / half_life);
}

void CostAwareReplacementPolicy::onAnchorTouched(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = resident_.find(anchor_id);
	if (it == resident_.end()) return;
	Clock::time_point now = Clock::now();
	it->second.heat = decayedHeat(it->second, now) + 1.0;
	it->second.last_update = now;
}

bool CostAwareReplacementPolicy::onRelocationTrigger() {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

bool CostAwareReplacementPolicy::hasPendingCandidates() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return !pending_candidates_.empty();
}

std::optional<PromotionCandidate> CostAwareReplacementPolicy::selectNextPromotionCandidate() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (pending_candidates_.empty()) return std::nullopt;
	PromotionCandidate candidate = std::move(pending_candidates_.front());
	pending_candidates_.pop_front();
	return candidate;
}

std::size_t CostAwareReplacementPolicy::roundedUnits(std::size_t bytes, std::size_t unit_bytes) {
	unit_bytes = std::max<std::size_t>(1, unit_bytes);
	return std::max<std::size_t>(1, (bytes + unit_bytes - 1) / unit_bytes);
}

double CostAwareReplacementPolicy::victimRetentionDensity(const ResidentEntry& entry,
		const EvictionCandidate& candidate, Clock::time_point now) const {
	double heat = decayedHeat(entry, now);
	double writeback_ratio = candidate.reclaimable_bytes == 0
														 ? 0.0
														 : static_cast<double>(candidate.potential_writeback_bytes) /
																 static_cast<double>(candidate.reclaimable_bytes);
	double cost = heat + config_.potential_writeback_weight * writeback_ratio;
	return cost / static_cast<double>(roundedUnits(candidate.reclaimable_bytes, 1));
}

AdmissionDecision CostAwareReplacementPolicy::evaluateAdmission(
		const PromotionCandidate& candidate, const AdmissionContext& context) {
	if (candidate.observations < config_.minimum_observations) return AdmissionDecision::Reject;
	if (config_.maximum_incremental_bytes != 0 &&
			context.incremental_bytes > config_.maximum_incremental_bytes) {
		return AdmissionDecision::Reject;
	}
	if (context.incremental_bytes == 0) return AdmissionDecision::Admit;

	std::size_t available = context.gpu_budget_bytes > context.gpu_bytes_allocated
													 ? context.gpu_budget_bytes - context.gpu_bytes_allocated
													 : 0;
	if (available >= context.incremental_bytes) return AdmissionDecision::Admit;

	std::lock_guard<std::mutex> lock(mutex_);
	Clock::time_point now = Clock::now();
	double best_victim_density = std::numeric_limits<double>::infinity();
	for (const EvictionCandidate& victim : context.eviction_candidates) {
		if (victim.anchor_id == candidate.anchor_id || victim.reclaimable_bytes == 0) continue;
		auto it = resident_.find(victim.anchor_id);
		if (it == resident_.end() || now - it->second.admitted_at < config_.minimum_residency) continue;
		best_victim_density = std::min(best_victim_density, victimRetentionDensity(it->second, victim, now));
	}
	if (!std::isfinite(best_victim_density)) return AdmissionDecision::Reject;

	double candidate_density = static_cast<double>(std::max<std::uint64_t>(1, candidate.observations)) /
													 static_cast<double>(roundedUnits(context.incremental_bytes,
																								 context.allocation_unit_bytes));
	// Victim density is expressed per byte; normalize the candidate's unit
	// density to the same scale before applying hysteresis.
	candidate_density /= static_cast<double>(std::max<std::size_t>(1, context.allocation_unit_bytes));
	return candidate_density >= best_victim_density * config_.admission_hysteresis
				 ? AdmissionDecision::Admit
				 : AdmissionDecision::Reject;
}

void CostAwareReplacementPolicy::onPromotionCommitted(VectorId anchor_id, const AdmissionContext& context) {
	std::lock_guard<std::mutex> lock(mutex_);
	Clock::time_point now = Clock::now();
	auto [it, inserted] = resident_.try_emplace(anchor_id, ResidentEntry{1.0, now, now, context.total_footprint_bytes});
	if (!inserted) {
		it->second.heat = decayedHeat(it->second, now) + 1.0;
		it->second.last_update = now;
		it->second.resident_bytes = context.total_footprint_bytes;
	}
}

std::optional<VectorId> CostAwareReplacementPolicy::selectEvictionCandidate(
		VectorId excluded, std::size_t required_bytes, const std::vector<EvictionCandidate>& candidates) {
	std::lock_guard<std::mutex> lock(mutex_);
	Clock::time_point now = Clock::now();
	double best_score = std::numeric_limits<double>::infinity();
	std::optional<VectorId> best;
	for (const EvictionCandidate& candidate : candidates) {
		if (candidate.anchor_id == excluded || candidate.reclaimable_bytes == 0) continue;
		auto it = resident_.find(candidate.anchor_id);
		if (it == resident_.end() || now - it->second.admitted_at < config_.minimum_residency) continue;
		double coverage_penalty = candidate.reclaimable_bytes >= required_bytes
														 ? 1.0
														 : static_cast<double>(required_bytes) /
																 static_cast<double>(candidate.reclaimable_bytes);
		double score = victimRetentionDensity(it->second, candidate, now) * coverage_penalty;
		if (score < best_score) {
			best_score = score;
			best = candidate.anchor_id;
		}
	}
	return best;
}

std::optional<VectorId> CostAwareReplacementPolicy::selectNextEvictionCandidate(VectorId excluded) {
	std::lock_guard<std::mutex> lock(mutex_);
	Clock::time_point now = Clock::now();
	double coldest = std::numeric_limits<double>::infinity();
	std::optional<VectorId> best;
	for (const auto& [anchor_id, entry] : resident_) {
		if (anchor_id == excluded || now - entry.admitted_at < config_.minimum_residency) continue;
		double heat = decayedHeat(entry, now);
		if (heat < coldest) {
			coldest = heat;
			best = anchor_id;
		}
	}
	return best;
}

}  // namespace arachne
