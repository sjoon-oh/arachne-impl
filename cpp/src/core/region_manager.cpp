#include "core/region_manager.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "gpu/dirty_header.hpp"
#include "logging.hpp"
#include "telemetry/trace.hpp"

namespace arachne {

// Implementation notes beyond region_manager.hpp's file-level overview:
//  - releaseAnchor() snapshots each Region *before* clearResidency()
//    overwrites the live record, then hands the snapshot (not a re-lookup)
//    to pending_reclaims_ -- so the Coordinator's later reclaim never races
//    a concurrent re-promotion of the same RegionId.
//  - coordinatorLoop() distinguishes a one-shot force-wake from a sticky
//    stop request: `forced` is consumed on the very next wakeup, but
//    coordinator_stop_requested_ stays true across every remaining
//    iteration, so shutdown() keeps forcing processPromotions() until the
//    policy's backlog is genuinely drained, not just once.
//  - processPromotions() flushes any Lease still pinned from earlier in the
//    same pass before every eviction attempt, since evictAnchorNow()'s
//    free() can await a Lease this same call chain already holds open.

namespace {
// Placeholder radius for a newly registered Anchor in RoutingCache
// (RoutingCache itself is radius-agnostic; each Anchor carries its own
// max_distance). A real per-query threshold is future work.
constexpr float kDefaultAnchorMaxDistance = 1e-3f;

std::size_t MinimumUtilizedBytes(std::size_t capacity, std::uint8_t percentage) {
	const std::size_t percent = std::min<std::size_t>(percentage, 100);
	const std::size_t whole = (capacity / 100) * percent;
	const std::size_t remainder = capacity % 100;
	return whole + (remainder * percent + 99) / 100;
}
}  // namespace

RegionManager::RegionManager(std::unique_ptr<ReplacementPolicy> replacement_policy)
		: replacement_policy_(std::move(replacement_policy)) {
	if (replacement_policy_ == nullptr) replacement_policy_ = std::make_unique<CostAwareReplacementPolicy>();
}

RegionManager::~RegionManager() { shutdown(); }

void RegionManager::registerRegion(RegionId id, HostRegionView host) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	if (regions_.find(id) != regions_.end()) return;

	Region region;
	region.id = id;
	region.host = host;
	regions_.emplace(id, region);
}

bool RegionManager::isRegistered(RegionId id) const {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	return regions_.find(id) != regions_.end();
}

Region RegionManager::regionOf(RegionId id) const {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it == regions_.end()) {
		throw std::invalid_argument("RegionManager: region is not registered");
	}
	return it->second;
}

std::vector<RegionId> RegionManager::regionsOf(VectorId anchor_id) const {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto it = dependencies_.find(anchor_id);
	if (it == dependencies_.end()) return {};
	return std::vector<RegionId>(it->second.begin(), it->second.end());
}

std::vector<RegionResidencyHint> RegionManager::residencyHints(VectorId anchor_id) const {
	// Diagnostic-only trace scope (ARACHNE_ENABLE_TRACING, zero-cost otherwise
	// -- see telemetry/trace.hpp): added to profile a timing gap between raw
	// hnswlib and Arachne's Controller stack, to check whether this
	// mutex-protected hot-path call is contributing. No behavior change.
	ARACHNE_TRACE_SCOPE("RegionManager", "residencyHints");
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto dependency_it = dependencies_.find(anchor_id);
	if (dependency_it == dependencies_.end() || dependency_it->second.empty()) return {};

	std::vector<RegionResidencyHint> hints;
	hints.reserve(dependency_it->second.size());
	for (RegionId region_id : dependency_it->second) {
		auto region_it = regions_.find(region_id);
		if (region_it == regions_.end() ||
				region_it->second.residency_state != RegionResidencyState::Resident ||
				!region_it->second.device.valid()) {
			return {};
		}
		hints.push_back({region_id, region_it->second.residency_generation});
	}
	return hints;
}

RegionManager::ResidencyPinBatch::~ResidencyPinBatch() {
	if (owner != nullptr) owner->unpinResidency(hints);
}

std::shared_ptr<void> RegionManager::tryPinResidency(
		const std::vector<RegionResidencyHint>& hints) {
	if (hints.empty()) return {};
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (const RegionResidencyHint& hint : hints) {
			auto it = regions_.find(hint.region);
			if (it == regions_.end() || it->second.residency_state != RegionResidencyState::Resident ||
					it->second.residency_generation != hint.generation || !it->second.device.valid()) {
				return {};
			}
		}
		for (const RegionResidencyHint& hint : hints) ++regions_.at(hint.region).residency_pins;
	}
	auto pin = std::make_shared<ResidencyPinBatch>();
	pin->owner = this;
	pin->hints = hints;
	return std::static_pointer_cast<void>(pin);
}

void RegionManager::unpinResidency(const std::vector<RegionResidencyHint>& hints) {
	bool reclaim_ready = false;
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (const RegionResidencyHint& hint : hints) {
			auto it = regions_.find(hint.region);
			if (it == regions_.end() || it->second.residency_pins == 0) continue;
			--it->second.residency_pins;
			if (it->second.residency_state == RegionResidencyState::Retiring &&
					it->second.residency_pins == 0) {
				reclaim_ready = true;
			}
		}
		if (reclaim_ready) coordinator_reclaim_ready_ = true;
	}
	if (reclaim_ready) coordinator_cv_.notify_one();
}

bool RegionManager::addDependency(VectorId anchor_id, RegionId region_id) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	if (regions_.find(region_id) == regions_.end()) return false;

	dependents_[region_id].insert(anchor_id);
	dependencies_[anchor_id].insert(region_id);
	return true;
}

bool RegionManager::removeDependency(VectorId anchor_id, RegionId region_id) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto dep_it = dependencies_.find(anchor_id);
	if (dep_it == dependencies_.end() || dep_it->second.erase(region_id) == 0) return false;
	if (dep_it->second.empty()) dependencies_.erase(dep_it);

	auto dependents_it = dependents_.find(region_id);
	if (dependents_it == dependents_.end()) return true;  // defensive: shouldn't happen

	dependents_it->second.erase(anchor_id);
	if (!dependents_it->second.empty()) return false;

	dependents_.erase(dependents_it);
	return true;
}

std::vector<RegionId> RegionManager::forget(VectorId anchor_id) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto dep_it = dependencies_.find(anchor_id);
	if (dep_it == dependencies_.end()) return {};

	std::vector<RegionId> orphaned;
	for (RegionId region_id : dep_it->second) {
		auto dependents_it = dependents_.find(region_id);
		if (dependents_it == dependents_.end()) continue;  // defensive: shouldn't happen

		dependents_it->second.erase(anchor_id);
		if (dependents_it->second.empty()) {
			dependents_.erase(dependents_it);
			orphaned.push_back(region_id);
		}
	}
	dependencies_.erase(dep_it);
	for (RegionId region_id : orphaned) region_group_.erase(region_id);

	// Leave anchor_id's group too, if it had one -- keeps group_members_ (and
	// therefore every remaining member's EvictionCandidate::group_members)
	// accurate immediately, rather than pointing at an Anchor that no longer
	// depends on anything. Reached both from releaseAnchor() (any
	// Controller-calling thread, via commitRemove()) and from group-based
	// eviction's own retireAnchorsNow() call (Coordinator thread only) --
	// already safe under mutex_, same as dependents_/dependencies_ above.
	if (auto group_it = anchor_group_.find(anchor_id); group_it != anchor_group_.end()) {
		GroupId group_id = group_it->second;
		anchor_group_.erase(group_it);
		if (auto members_it = group_members_.find(group_id); members_it != group_members_.end()) {
			members_it->second.erase(anchor_id);
			if (members_it->second.empty()) group_members_.erase(members_it);
		}
	}
	return orphaned;
}

void RegionManager::setLease(RegionId id, LeaseHandle lease) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it != regions_.end()) it->second.lease = lease;
}

void RegionManager::setDevice(RegionId id, gpu::DeviceRegionHandle device) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it != regions_.end()) it->second.device = device;
}

void RegionManager::clearResidency(RegionId id) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it == regions_.end()) return;

	it->second.lease = LeaseHandle{};
	it->second.device = gpu::DeviceRegionHandle{};
	it->second.residency_state = RegionResidencyState::HostOnly;
	++it->second.residency_generation;
}

// ---------------------------------------------------------------------------
// Coordinator lifecycle
// ---------------------------------------------------------------------------

void RegionManager::start(IAdapter& adapter, gpu::DeviceRegionPool& device_region_pool, RoutingCache& routing_cache,
											 CoordinatorConfig config) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	if (coordinator_running_) {
		throw std::logic_error("RegionManager coordinator already started");
	}
	adapter_ = &adapter;
	device_region_pool_ = &device_region_pool;
	routing_cache_ = &routing_cache;
	config.near_fit_min_utilization_percent =
			std::min<std::uint8_t>(config.near_fit_min_utilization_percent, 100);
	coordinator_config_ = config;
	coordinator_stop_requested_ = false;
	coordinator_running_ = true;
	coordinator_ = std::thread(&RegionManager::coordinatorLoop, this);
}

void RegionManager::shutdown() {
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		if (!coordinator_running_) return;
		coordinator_stop_requested_ = true;
		coordinator_force_wake_ = true;
	}
	coordinator_cv_.notify_all();

	if (coordinator_.joinable()) coordinator_.join();

	std::lock_guard<RegionManagerMutex> lock(mutex_);
	coordinator_running_ = false;
	coordinator_stop_requested_ = false;
	adapter_ = nullptr;
	device_region_pool_ = nullptr;
	routing_cache_ = nullptr;
}

std::uint64_t RegionManager::currentEpochLocked(VectorId anchor_id) const {
	auto it = anchor_epoch_.find(anchor_id);
	return it == anchor_epoch_.end() ? 0 : it->second;
}

void RegionManager::requestPromotion(VectorId anchor_id, RegionFootprint footprint, VectorView vector) {
	ARACHNE_TRACE_SCOPE("RegionManager", "requestPromotion");
	// Pure enqueue -- no replacement_policy_ interaction here; only the
	// Coordinator thread hands candidates to the policy (see the header's
	// file-level overview).
	PromotionCandidate candidate;
	candidate.anchor_id = anchor_id;
	candidate.footprint = std::move(footprint);
	candidate.vector_dim = vector.dim;
	candidate.vector_dtype = vector.dtype;
	candidate.enqueued_at = PromotionCandidate::Clock::now();
	candidate.enqueue_sequence = next_candidate_sequence_.fetch_add(1, std::memory_order_relaxed);
	if (vector.data != nullptr && vector.dim > 0) {
		const auto* bytes = static_cast<const std::byte*>(vector.data);
		candidate.vector_bytes.assign(bytes, bytes + static_cast<std::size_t>(vector.dim) * VectorElementSize(vector.dtype));
	}

	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		candidate.epoch = currentEpochLocked(anchor_id);
		pending_promotions_.push_back(std::move(candidate));
	}
	// Event-driven MPSC intake: wake the single consumer immediately. The
	// relocation itself still waits for the coalescing deadline.
	coordinator_cv_.notify_one();
}

void RegionManager::releaseAnchor(VectorId anchor_id) {
	ARACHNE_TRACE_SCOPE("RegionManager", "releaseAnchor");
	// Synchronous: dependency bookkeeping, lease release, and clearResidency()
	// all happen here so an orphaned Region is instantly re-promotable for a
	// different Anchor, without racing this Region's still-pending reclaim
	// below (captured into an independent snapshot, never looked up live again).
	std::vector<RegionId> orphaned = forget(anchor_id);

	// Must stop tracking anchor_id regardless of whether forget() freed a
	// Region: an Anchor evicted from a still-multiply-depended-on Region must
	// still stop being FIFO-selectable, or a capacity-retry loop elsewhere
	// could re-select it forever without making progress.
	replacement_policy_->onAnchorEvicted(anchor_id);

	// Bump the epoch *before* erasing from RoutingCache -- any
	// PromotionCandidate for anchor_id enqueued before this call now carries
	// a stale epoch and will be discarded by processPromotions().
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		++anchor_epoch_[anchor_id];
	}
	if (routing_cache_ != nullptr) routing_cache_->erase(anchor_id);

	if (orphaned.empty()) return;

	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (RegionId region_id : orphaned) {
			auto it = regions_.find(region_id);
			if (it == regions_.end() || it->second.residency_state != RegionResidencyState::Resident) continue;
			it->second.residency_state = RegionResidencyState::Retiring;
			++it->second.residency_generation;
			pending_reclaims_.push_back(it->second);
		}
		coordinator_reclaim_ready_ = true;
	}
	stat_anchor_evictions_.fetch_add(1, std::memory_order_relaxed);
	coordinator_cv_.notify_one();
}

bool RegionManager::isKnownAnchor(VectorId anchor_id) const {
	ARACHNE_TRACE_SCOPE("RegionManager", "isKnownAnchor");
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	return dependencies_.contains(anchor_id) || anchor_epoch_.contains(anchor_id);
}

void RegionManager::recordTraversal(const RegionFootprint& touched) {
	ARACHNE_TRACE_SCOPE("RegionManager", "recordTraversal");
	std::unordered_set<VectorId> anchors;
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (RegionId region_id : touched.regions) {
			auto it = dependents_.find(region_id);
			if (it == dependents_.end()) continue;  // unregistered, or currently orphaned
			anchors.insert(it->second.begin(), it->second.end());
		}
	}
	for (VectorId anchor_id : anchors) replacement_policy_->onAnchorTouched(anchor_id);
}

void RegionManager::waitIdle() {
	std::unique_lock<RegionManagerMutex> lock(mutex_);
	coordinator_force_wake_ = true;
	lock.unlock();
	coordinator_cv_.notify_all();

	lock.lock();
	idle_cv_.wait(lock, [this] {
		// Empty intake queues only prove they've been drained *into* the
		// policy, not that the policy has offered every admitted candidate
		// back out yet -- hasPendingCandidates() must also be checked, or this
		// could return before every requested promotion was actually attempted.
		return !coordinator_busy_ && pending_promotions_.empty() && pending_reclaims_.empty() &&
					 !replacement_policy_->hasPendingCandidates();
	});
}

RegionManager::Stats RegionManager::stats() const {
	Stats result;
	result.regions_promoted_total = stat_regions_promoted_.load(std::memory_order_relaxed);
	result.regions_evicted_total = stat_regions_evicted_.load(std::memory_order_relaxed);
	result.regions_written_back_total = stat_regions_written_back_.load(std::memory_order_relaxed);
	result.anchor_evictions_total = stat_anchor_evictions_.load(std::memory_order_relaxed);
	result.compactions_total = stat_compactions_total_.load(std::memory_order_relaxed);
	result.relocation_batches_total = stat_relocation_batches_.load(std::memory_order_relaxed);
	result.candidates_requeued_total = stat_candidates_requeued_.load(std::memory_order_relaxed);
	result.near_fit_reuses_total = stat_near_fit_reuses_.load(std::memory_order_relaxed);
	result.candidates_rejected_total = stat_candidates_rejected_.load(std::memory_order_relaxed);
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	if (device_region_pool_ != nullptr) result.gpu_bytes_allocated = device_region_pool_->bytesAllocated();
	return result;
}

void RegionManager::coordinatorLoop() {
	using Clock = std::chrono::steady_clock;
	std::optional<Clock::time_point> relocation_deadline;

	while (true) {
		std::unique_lock<RegionManagerMutex> lock(mutex_);
		auto wake_requested = [this] {
			return coordinator_stop_requested_ || coordinator_force_wake_ || coordinator_reclaim_ready_ ||
					 !pending_promotions_.empty();
		};
		if (relocation_deadline.has_value()) {
			coordinator_cv_.wait_until(lock, *relocation_deadline, wake_requested);
		} else {
			coordinator_cv_.wait(lock, wake_requested);
		}
		const Clock::time_point now = Clock::now();
		const bool deadline_reached = relocation_deadline.has_value() && now >= *relocation_deadline;
		bool forced = coordinator_force_wake_;
		coordinator_force_wake_ = false;
		bool reclaim_ready = coordinator_reclaim_ready_;
		coordinator_reclaim_ready_ = false;
		bool stop = coordinator_stop_requested_;

		coordinator_busy_ = true;
		std::vector<PromotionCandidate> admitted(std::make_move_iterator(pending_promotions_.begin()),
																							std::make_move_iterator(pending_promotions_.end()));
		pending_promotions_.clear();
		std::vector<Region> reclaims(std::make_move_iterator(pending_reclaims_.begin()),
																 std::make_move_iterator(pending_reclaims_.end()));
		pending_reclaims_.clear();
		lock.unlock();

		ARACHNE_LOG_INFO(
				"coordinatorLoop: wakeup -- forced={} stop={} deadline_reached={} reclaim_ready={} admitted={} "
				"reclaims={}",
				forced, stop, deadline_reached, reclaim_ready, admitted.size(), reclaims.size());

		// Ingestion happens every wakeup regardless of what runs below -- this
		// keeps the policy's view current without waiting for a trigger.
		for (PromotionCandidate& candidate : admitted) {
			replacement_policy_->enqueueCandidate(std::move(candidate));
		}
		if (!admitted.empty() && !relocation_deadline.has_value()) {
			relocation_deadline = now + coordinator_config_.trigger_interval;
		}

		// Reclaims are already-decided cleanup (from releaseAnchor()/
		// evictAnchorNow()), not a policy decision -- process unconditionally.
		if (!reclaims.empty()) reclaimRegions(reclaims);

		// Execution is gated by the policy's own onRelocationTrigger(), unless
		// force-woken or stopping (both must guarantee every admitted
		// candidate is actually offered). `stop` is checked separately from
		// `forced` because `forced` is one-shot but stop stays true across
		// every remaining iteration, so shutdown() keeps forcing execution
		// until the policy's backlog is genuinely drained (see below).
		const bool execution_point = forced || stop || deadline_reached || reclaim_ready;
		if (execution_point && replacement_policy_->hasPendingCandidates() &&
				(forced || stop || replacement_policy_->onRelocationTrigger())) {
			// A forced drain must terminate: transient failures are attempted once
			// and then dropped, matching waitIdle()/shutdown()'s liveness contract.
			do {
				processRelocationBatch(/*retain_failed_candidates=*/!forced && !stop);
			} while ((forced || stop) && replacement_policy_->hasPendingCandidates());
			relocation_deadline.reset();
		}
		if (replacement_policy_->hasPendingCandidates() && !relocation_deadline.has_value()) {
			relocation_deadline = Clock::now() + coordinator_config_.trigger_interval;
		}

		lock.lock();
		coordinator_busy_ = false;
		idle_cv_.notify_all();
		if (stop && pending_promotions_.empty() && pending_reclaims_.empty() &&
				!replacement_policy_->hasPendingCandidates()) {
			break;
		}
	}
}

// ---------------------------------------------------------------------------
// GPU residency management -- relocated from Controller, unchanged in
// behavior except for what triggers each step (batched Coordinator pass
// instead of one Controller-calling thread per Anchor).
// ---------------------------------------------------------------------------

std::optional<gpu::DeviceRegionHandle> RegionManager::allocateWithCompaction(
		std::size_t bytes, gpu::DeviceRegionPool::TransferBatch& pending) {
	std::optional<gpu::DeviceRegionHandle> handle = device_region_pool_->tryAllocate(bytes);
	if (handle.has_value()) return handle;

	if (!device_region_pool_->hasCapacity(bytes)) {
		return std::nullopt;  // genuinely over budget -- compacting frees nothing new
	}

	// Flush and release any Lease already pinned earlier in this batch first:
	// compact() only relocates unpinned allocations, so this widens its pool
	// of movable blocks instead of excluding Regions this same batch pinned.
	if (!pending.leases.empty()) {
		device_region_pool_->finishTransfers(pending);
	}
	device_region_pool_->compact(gpu::MemoryKind::Data, bytes);
	stat_compactions_total_.fetch_add(1, std::memory_order_relaxed);
	return device_region_pool_->tryAllocate(bytes);
}

RegionManager::MakeResult RegionManager::make(VectorId anchor_id, std::uint64_t anchor_epoch, RegionId region,
		gpu::DeviceRegionPool::TransferBatch& pending,
		std::vector<PendingPromotionCommit>& commits, ReusableAllocations& reusable) {
	ARACHNE_TRACE_SCOPE("RegionManager", "make");
	for (RegionId existing : regionsOf(anchor_id)) {
		if (existing == region) return MakeResult::Promoted;  // already a dependent
	}

	if (!isRegistered(region)) {
		ARACHNE_LOG_DEBUG("make: region {} is not registered, anchor {} cannot depend on it", region, anchor_id);
		return MakeResult::NotEligible;
	}

	Region snapshot = regionOf(region);
	if (snapshot.residency_state == RegionResidencyState::HostOnly) {
		IRegion* target = adapter_->resolveRegion(region);
		if (target == nullptr) return MakeResult::NotEligible;

		LeaseHandle lease = target->acquireWriteLease();
		if (!lease.valid()) {
			ARACHNE_LOG_DEBUG("make: region {} not lease-eligible for anchor {}", region, anchor_id);
			return MakeResult::NotEligible;
		}

		std::size_t header_bytes = gpu::DirtyHeaderBytes(snapshot.host.bytes, snapshot.host.subregion_bytes);
		std::optional<gpu::DeviceRegionHandle> device;
		auto reusable_it = reusable.find(region);
		if (reusable_it != reusable.end()) {
			device = reusable_it->second;
			reusable.erase(reusable_it);
		} else {
			device = allocateWithCompaction(header_bytes + snapshot.host.bytes, pending);
		}
		if (!device.has_value()) {
			target->releaseWriteLease(lease);
			ARACHNE_LOG_DEBUG("make: region {} ({} bytes) out of GPU capacity for anchor {}", region,
												 header_bytes + snapshot.host.bytes, anchor_id);
			return MakeResult::OutOfCapacity;
		}
		std::uint64_t residency_generation = 0;
		{
			std::lock_guard<RegionManagerMutex> lock(mutex_);
			auto live = regions_.find(region);
			if (live == regions_.end() ||
					live->second.residency_state != RegionResidencyState::HostOnly ||
					currentEpochLocked(anchor_id) != anchor_epoch) {
				target->releaseWriteLease(lease);
				device_region_pool_->free(*device);
				return MakeResult::NotEligible;
			}
			live->second.residency_state = RegionResidencyState::Promoting;
			residency_generation = ++live->second.residency_generation;
		}
		if (header_bytes > 0) {
			std::vector<std::byte> zero_header(header_bytes, std::byte{0});
			device_region_pool_->enqueueCopyFromHost(*device, zero_header.data(), header_bytes,
																								/*dst_offset=*/0, pending);
		}
		device_region_pool_->enqueueCopyFromHost(*device, snapshot.host.ptr, snapshot.host.bytes, header_bytes,
																							pending);
		commits.push_back(
				PendingPromotionCommit{anchor_id, region, *device, lease, residency_generation, anchor_epoch});
		return MakeResult::Promoted;
	}
	if (snapshot.residency_state == RegionResidencyState::Promoting ||
			snapshot.residency_state == RegionResidencyState::Retiring) {
		return MakeResult::Deferred;
	}
	if (snapshot.residency_state != RegionResidencyState::Resident || !snapshot.lease.valid()) {
		return MakeResult::NotEligible;
	}

	// No replacement_policy_ notification here -- selectNextPromotionCandidate()
	// already recorded anchor_id for eviction-ordering purposes before make()
	// was ever called for it.
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		auto live = regions_.find(region);
		if (live == regions_.end() || live->second.residency_state != RegionResidencyState::Resident ||
				currentEpochLocked(anchor_id) != anchor_epoch) {
			return MakeResult::Deferred;
		}
		dependents_[region].insert(anchor_id);
		dependencies_[anchor_id].insert(region);
	}

	ARACHNE_LOG_DEBUG("make: anchor {} now depends on region {}", anchor_id, region);
	return MakeResult::Promoted;
}

std::size_t RegionManager::reservedRegionBytes(const Region& region) const {
	std::size_t logical_bytes = gpu::DirtyHeaderBytes(region.host.bytes, region.host.subregion_bytes) +
														region.host.bytes;
	return device_region_pool_->reservationBytes(logical_bytes, gpu::MemoryKind::Data);
}

std::vector<EvictionCandidate> RegionManager::buildEvictionCandidates() const {
	// Diagnostic-only (ARACHNE_ENABLE_TRACING build) -- see the latency-
	// tracing report entry this was added for.
	ARACHNE_TRACE_SCOPE("RegionManager", "buildEvictionCandidates");
	std::vector<EvictionCandidate> candidates;
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	candidates.reserve(dependencies_.size());
	for (const auto& [anchor_id, region_ids] : dependencies_) {
		EvictionCandidate candidate;
		candidate.anchor_id = anchor_id;

		// This Anchor's group (see assignAnchorToGroup()'s doc comment) --
		// falls back to a solo {anchor_id} set for an Anchor that was never
		// assigned one (predates group tracking, or simply hasn't been through
		// a promotion commit yet), which reproduces the original
		// sole-ownership-only rule below exactly.
		const std::unordered_set<VectorId>* members = nullptr;
		std::unordered_set<VectorId> solo;
		if (auto group_it = anchor_group_.find(anchor_id); group_it != anchor_group_.end()) {
			if (auto members_it = group_members_.find(group_it->second); members_it != group_members_.end()) {
				members = &members_it->second;
			}
		}
		if (members == nullptr) {
			solo.insert(anchor_id);
			members = &solo;
		}
		candidate.group_members.assign(members->begin(), members->end());

		for (RegionId region_id : region_ids) {
			auto region_it = regions_.find(region_id);
			if (region_it == regions_.end() || !region_it->second.device.valid()) continue;
			const Region& region = region_it->second;
			std::size_t bytes = reservedRegionBytes(region);
			candidate.resident_bytes += bytes;
			++candidate.resident_regions;
			auto dependents_it = dependents_.find(region_id);
			// Reclaimable by evicting this Anchor's whole *group* together --
			// every current dependent of the Region must be a group member, not
			// just anchor_id itself (the group_merge_overlap_threshold==1-member
			// default makes this identical to the original "exactly one
			// dependent" check).
			bool reclaimable_by_group = dependents_it != dependents_.end() &&
					std::all_of(dependents_it->second.begin(), dependents_it->second.end(),
							[members](VectorId a) { return members->count(a) != 0; });
			if (reclaimable_by_group) {
				candidate.reclaimable_bytes += bytes;
				if (region.residency_state == RegionResidencyState::Resident && region.residency_pins == 0) {
					candidate.reclaimable_now_bytes += bytes;
				}
				candidate.potential_writeback_bytes += region.host.bytes;
				++candidate.reclaimable_regions;
			}
		}
		if (candidate.resident_regions != 0) candidates.push_back(std::move(candidate));
	}
	return candidates;
}

void RegionManager::assignAnchorToGroup(VectorId anchor_id, const std::vector<RegionId>& footprint_regions) {
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	if (auto existing = anchor_group_.find(anchor_id); existing != anchor_group_.end()) {
		// Already grouped (e.g. a re-promotion re-confirming residency) --
		// leave it in place, just extend the group's region tags so a *future*
		// candidate's overlap computation sees this footprint too.
		for (RegionId region_id : footprint_regions) region_group_[region_id] = existing->second;
		return;
	}

	std::unordered_map<GroupId, std::size_t> overlap_count;
	for (RegionId region_id : footprint_regions) {
		if (auto it = region_group_.find(region_id); it != region_group_.end()) ++overlap_count[it->second];
	}
	GroupId best_group = 0;
	std::size_t best_overlap = 0;
	for (const auto& [group_id, count] : overlap_count) {
		if (count > best_overlap) {
			best_overlap = count;
			best_group = group_id;
		}
	}
	double overlap_ratio = footprint_regions.empty()
			? 0.0
			: static_cast<double>(best_overlap) / static_cast<double>(footprint_regions.size());

	GroupId assigned;
	auto best_members_it = best_group != 0 ? group_members_.find(best_group) : group_members_.end();
	bool can_join = best_group != 0 && overlap_ratio >= coordinator_config_.group_merge_overlap_threshold &&
			best_members_it != group_members_.end() &&
			best_members_it->second.size() < coordinator_config_.max_eviction_group_size;
	if (can_join) {
		assigned = best_group;
	} else {
		assigned = next_group_id_++;
	}
	group_members_[assigned].insert(anchor_id);
	anchor_group_[anchor_id] = assigned;
	for (RegionId region_id : footprint_regions) region_group_[region_id] = assigned;
}

AdmissionContext RegionManager::buildAdmissionContext(const PromotionCandidate& candidate,
		std::shared_ptr<const std::vector<EvictionCandidate>>& eviction_candidates_cache) const {
	// Diagnostic-only (ARACHNE_ENABLE_TRACING build) -- see the 10M-scale
	// Coordinator-throughput investigation this was added for
	// (cpp/test/index/report/): buildRelocationPlan_collect's own per-pass
	// scope was already traced, but nothing isolated this specific callee's
	// own cost from the rest of that loop body.
	ARACHNE_TRACE_SCOPE("RegionManager", "buildAdmissionContext");
	AdmissionContext context;
	context.allocation_unit_bytes = device_region_pool_->allocationUnitBytes(gpu::MemoryKind::Data);
	std::unordered_set<RegionId> unique_regions;
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (RegionId region_id : candidate.footprint.regions) {
			if (!unique_regions.insert(region_id).second) continue;
			auto it = regions_.find(region_id);
			if (it == regions_.end()) continue;
			std::size_t bytes = reservedRegionBytes(it->second);
			context.total_footprint_bytes += bytes;
			if (it->second.device.valid()) {
				context.already_resident_bytes += bytes;
			} else {
				context.incremental_bytes += bytes;
			}
		}
	}
	context.gpu_bytes_allocated = device_region_pool_->bytesReserved(gpu::MemoryKind::Data);
	context.gpu_budget_bytes = device_region_pool_->budgetBytes(gpu::MemoryKind::Data);

	// buildEvictionCandidates() is a real cost (scans every tracked Anchor's
	// dependencies while holding mutex_ -- the same mutex_ hot-path calls
	// like tryPinResidency() need, so an unnecessary scan here is contended
	// lock time for search()/insert() callers, not just wasted Coordinator-
	// thread work). Skip it entirely when there's already enough free
	// capacity to admit this candidate without evicting anything -- no
	// policy's evaluateAdmission()/evaluateBatchAdmission() can act on
	// eviction_candidates before checking that anyway (see AdmissionContext's
	// own doc comment). When eviction info *is* needed, compute it at most
	// once per buildRelocationPlan() pass and reuse the cached result for
	// every other candidate examined in that same pass -- see this method's
	// declaration (region_manager.hpp) for why that's always exactly as
	// fresh as recomputing it every time.
	std::size_t available =
			context.gpu_budget_bytes > context.gpu_bytes_allocated ? context.gpu_budget_bytes - context.gpu_bytes_allocated : 0;
	if (available < context.incremental_bytes) {
		if (!eviction_candidates_cache) {
			eviction_candidates_cache = std::make_shared<const std::vector<EvictionCandidate>>(buildEvictionCandidates());
		}
		// Shares the cached snapshot itself (an atomic refcount bump), not a
		// deep copy of it -- see AdmissionContext::eviction_candidates' own
		// doc comment for why an owned std::vector here was the actual cost
		// of this whole function once it's called once per *candidate*
		// rather than once per pass.
		context.eviction_candidates = eviction_candidates_cache;
	}
	return context;
}

std::size_t RegionManager::promotionBytes(const std::vector<PlannedPromotion>& promotions) const {
	std::unordered_set<RegionId> unique;
	std::size_t bytes = 0;
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	for (const PlannedPromotion& promotion : promotions) {
		for (RegionId region_id : promotion.candidate.footprint.regions) {
			if (!unique.insert(region_id).second) continue;
			auto it = regions_.find(region_id);
			if (it == regions_.end() || it->second.residency_state != RegionResidencyState::HostOnly) continue;
			bytes += reservedRegionBytes(it->second);
		}
	}
	return bytes;
}

std::size_t RegionManager::projectedReclaimableBytes(
		const std::vector<VectorId>& victims, bool require_unpinned) const {
	std::unordered_set<VectorId> selected(victims.begin(), victims.end());
	std::size_t bytes = 0;
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	for (const auto& [region_id, anchors] : dependents_) {
		if (anchors.empty() || !std::all_of(anchors.begin(), anchors.end(),
				[&selected](VectorId anchor) { return selected.contains(anchor); })) {
			continue;
		}
		auto it = regions_.find(region_id);
		if (it == regions_.end() || it->second.residency_state != RegionResidencyState::Resident ||
				!it->second.device.valid() || (require_unpinned && it->second.residency_pins != 0)) {
			continue;
		}
		bytes += reservedRegionBytes(it->second);
	}
	return bytes;
}

std::vector<RegionManager::PromotionStorageRequest> RegionManager::buildPromotionStorageRequests(
		const std::vector<PlannedPromotion>& promotions) const {
	std::vector<PromotionStorageRequest> requests;
	std::unordered_set<RegionId> unique;
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	for (const PlannedPromotion& promotion : promotions) {
		for (RegionId region_id : promotion.candidate.footprint.regions) {
			if (!unique.insert(region_id).second) continue;
			auto it = regions_.find(region_id);
			if (it == regions_.end() || it->second.residency_state != RegionResidencyState::HostOnly) continue;
			std::size_t logical = gpu::DirtyHeaderBytes(it->second.host.bytes, it->second.host.subregion_bytes) +
					it->second.host.bytes;
			requests.push_back({region_id, logical, reservedRegionBytes(it->second)});
		}
	}
	return requests;
}

void RegionManager::requeueCandidates(std::vector<PromotionCandidate> candidates) {
	stat_candidates_requeued_.fetch_add(candidates.size(), std::memory_order_relaxed);
	for (PromotionCandidate& candidate : candidates) {
		replacement_policy_->requeueCandidate(std::move(candidate));
	}
}

std::optional<RegionManager::RelocationPlan> RegionManager::buildRelocationPlan(
		std::uint64_t batch_sequence, bool retain_failed_candidates) {
	RelocationPlan plan;
	plan.batch_sequence = batch_sequence;
	std::vector<PromotionCandidate> deferred;
	const std::size_t budget = device_region_pool_->budgetBytes(gpu::MemoryKind::Data);
	const std::size_t reserved = device_region_pool_->bytesReserved(gpu::MemoryKind::Data);
	const std::size_t available = budget > reserved ? budget - reserved : 0;
	std::size_t selected_incremental = 0;
	// Populated lazily by buildAdmissionContext() the first time any
	// candidate in this pass actually needs eviction info, then reused for
	// every other candidate examined here and for the victim-selection loop
	// below -- see buildAdmissionContext()'s own comment for why recomputing
	// it more than once per pass is always redundant, and
	// AdmissionContext::eviction_candidates' own doc comment for why this is
	// a shared_ptr rather than a plain std::optional<std::vector<...>>.
	std::shared_ptr<const std::vector<EvictionCandidate>> eviction_candidates_cache;

	// Diagnostic-only (ARACHNE_ENABLE_TRACING build): brace-scoped so the
	// timer covers exactly the promotion-collection loop below, distinct
	// from the victim-selection loop further down -- see the latency-tracing
	// report entry this was added for.
	{
	ARACHNE_TRACE_SCOPE("RegionManager", "buildRelocationPlan_collect");
	// Diagnostic-only (ARACHNE_ENABLE_TRACING build) -- see the pass-count
	// investigation this was added for (cpp/test/index/report/): a per-PASS
	// (not per-candidate, so low volume -- one line per buildRelocationPlan_collect
	// call, unlike the existing per-candidate log a few lines below) summary of
	// how many candidates this pass's collect loop actually looked at and why
	// it stopped, to distinguish "queue genuinely ran dry" from "broke out
	// early because of the per-pass budget cap" as the reason a big backlog
	// needs many separate passes to drain.
	std::size_t examined_this_pass = 0;
	while (true) {
		std::optional<PromotionCandidate> candidate = replacement_policy_->selectNextPromotionCandidate();
		if (!candidate.has_value()) {
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: collect loop ended -- reason=queue_drained examined={} admitted={} "
					"retain_failed_candidates={}",
					batch_sequence, examined_this_pass, plan.promotions.size(), retain_failed_candidates);
			break;
		}
		++examined_this_pass;
		{
			std::lock_guard<RegionManagerMutex> lock(mutex_);
			if (currentEpochLocked(candidate->anchor_id) != candidate->epoch) continue;
		}
		++candidate->planning_attempts;
		candidate->last_batch_sequence = batch_sequence;
		if (candidate->first_batch_sequence == 0) candidate->first_batch_sequence = batch_sequence;

		AdmissionContext admission = buildAdmissionContext(*candidate, eviction_candidates_cache);
		RelocationBatchContext batch_context{batch_sequence, plan.promotions.size(), selected_incremental,
				available, budget, coordinator_config_.max_promotion_bytes_per_pass};
		BatchAdmissionDecision decision =
				replacement_policy_->evaluateBatchAdmission(*candidate, admission, batch_context);
		{
			// Diagnostic-only (ARACHNE_ENABLE_TRACING build) -- see the
			// 10M-scale Coordinator-throughput investigation this was added
			// for (cpp/test/index/report/): isolates ARACHNE_LOG_INFO's own
			// cost (fmt::format() runs unconditionally as a macro argument,
			// before the logger's own runtime level check ever sees it --
			// see logging.hpp -- so --quiet-logs suppresses the *output*,
			// not this formatting cost) from the rest of this loop body.
			ARACHNE_TRACE_SCOPE("RegionManager", "buildRelocationPlan_collect_log");
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: admission for anchor {} (incremental_bytes={}) -> {} "
					"(plan.promotions.size()={} selected_incremental={} retain_failed_candidates={})",
					batch_sequence, candidate->anchor_id, admission.incremental_bytes,
					decision == BatchAdmissionDecision::Admit ? "Admit"
					: decision == BatchAdmissionDecision::Defer ? "Defer"
																											 : "Reject",
					plan.promotions.size(), selected_incremental, retain_failed_candidates);
		}
		if (decision == BatchAdmissionDecision::Reject) {
			stat_candidates_rejected_.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		if (decision == BatchAdmissionDecision::Defer) {
			if (retain_failed_candidates) deferred.push_back(std::move(*candidate));
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: collect loop ended -- reason=defer examined={} admitted={} "
					"retain_failed_candidates={}",
					batch_sequence, examined_this_pass, plan.promotions.size(), retain_failed_candidates);
			break;
		}

		if (coordinator_config_.max_promotion_bytes_per_pass != 0 && !plan.promotions.empty() &&
				(admission.incremental_bytes > coordinator_config_.max_promotion_bytes_per_pass ||
				 selected_incremental > coordinator_config_.max_promotion_bytes_per_pass - admission.incremental_bytes)) {
			// Always requeue here, independent of retain_failed_candidates -- see
			// the hard-budget branch below for the termination argument this
			// relies on (plan.promotions non-empty here is exactly the condition
			// that makes it safe): this pass already admitted at least one
			// candidate before this one hit the cap, so retrying it on a later
			// pass can only make progress, never spin. Matches CoordinatorConfig::
			// max_promotion_bytes_per_pass's own doc comment ("subsequent work is
			// returned to the policy for a later pass") -- that was always the
			// intent, not something conditional on how the pass was triggered.
			deferred.push_back(std::move(*candidate));
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: collect loop ended -- reason=max_promotion_bytes_per_pass examined={} "
					"admitted={} retain_failed_candidates={}",
					batch_sequence, examined_this_pass, plan.promotions.size(), retain_failed_candidates);
			break;
		}

		plan.promotions.push_back(PlannedPromotion{std::move(*candidate), std::move(admission)});
		std::size_t required = promotionBytes(plan.promotions);
		if (required > budget) {
			VectorId dropped_anchor = plan.promotions.back().candidate.anchor_id;
			PromotionCandidate over_budget = std::move(plan.promotions.back().candidate);
			plan.promotions.pop_back();
			// Requeue whenever something else already claimed room in this same
			// pass (plan.promotions non-empty after the pop) -- independent of
			// retain_failed_candidates, and safe even during a forced/stop drain:
			// every pass's *first* offered candidate is admitted unconditionally
			// (this branch can only fire once plan.promotions already holds an
			// earlier admission, or this candidate stands entirely alone -- see
			// below), so replacement_policy_'s pending set shrinks by at least
			// one candidate every coordinatorLoop() do-while iteration regardless
			// of which case fires -- the loop still terminates in a bounded
			// number of passes (at most the number of distinct candidates
			// pending when the drain started).
			//
			// A candidate that's over budget entirely *by itself* (plan.promotions
			// empty here -- it was this pass's first and only entry) is a
			// genuinely different, permanent case: no amount of eviction could
			// ever make a single candidate larger than the whole budget fit, so
			// it's dropped for good rather than requeued, regardless of
			// retain_failed_candidates -- retrying it would just spin forever.
			bool requeued = !plan.promotions.empty();
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: anchor {} pushed cumulative required={} over budget={} with {} other "
					"promotion(s) already in this batch -- {}",
					batch_sequence, dropped_anchor, required, budget, plan.promotions.size(),
					requeued ? "requeued for a later pass" : "DROPPED PERMANENTLY (infeasible alone, not requeued)");
			if (requeued) deferred.push_back(std::move(over_budget));
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: collect loop ended -- reason=over_budget examined={} admitted={} "
					"required={} budget={} retain_failed_candidates={}",
					batch_sequence, examined_this_pass, plan.promotions.size(), required, budget,
					retain_failed_candidates);
			break;
		}
		selected_incremental += plan.promotions.back().admission.incremental_bytes;
	}
	}  // end buildRelocationPlan_collect trace scope

	if (!deferred.empty()) requeueCandidates(std::move(deferred));
	if (plan.promotions.empty()) return std::nullopt;
	plan.required_incremental_bytes = promotionBytes(plan.promotions);
	if (available >= plan.required_incremental_bytes) return plan;

	// Reuses eviction_candidates_cache if some candidate above already
	// populated it -- same "static for the whole pass" reasoning as
	// buildAdmissionContext()'s own use of it.
	if (!eviction_candidates_cache) {
		eviction_candidates_cache = std::make_shared<const std::vector<EvictionCandidate>>(buildEvictionCandidates());
	}
	// excluded_anchors starts as "already promoted this pass" (never evict the
	// thing just promoted) and grows below as each victim group is selected --
	// deliberately NOT expressed as a filtered copy of *eviction_candidates_cache
	// (that used to be this exact spot: `std::vector<EvictionCandidate>
	// candidates = *eviction_candidates_cache;` then erase-remove). That copy
	// was the dominant Coordinator cost at 10M scale -- eviction_candidates_cache
	// can hold thousands of entries, each with its own heap-allocated
	// group_members, and the old code deep-copied the *whole thing* once per
	// pass just to physically remove a handful of already-decided anchors; at
	// 10M scale, pass count itself is 44-250x what it is at 1M scale, so this
	// added up to the majority of a 68-minute post-loop stall -- see
	// cpp/test/index/report/2026-08-31-10m-scale-coordinator-throughput.md and
	// its follow-up fix entry. eviction_candidates_cache itself is now shared
	// (no copy) with every selectEvictionCandidate() call below; excluded_anchors
	// is the small, mutable side-table that expresses "skip this one" instead --
	// see ReplacementPolicy::selectEvictionCandidate()'s own doc comment for the
	// exclusion contract every policy must honor against it.
	std::unordered_set<VectorId> excluded_anchors;
	for (const PlannedPromotion& promotion : plan.promotions) {
		excluded_anchors.insert(promotion.candidate.anchor_id);
	}
	ARACHNE_LOG_INFO(
			"buildRelocationPlan[{}]: required={} available={} eviction_candidates={} (excluded so far: {}) -- "
			"entering victim-selection loop",
			batch_sequence, plan.required_incremental_bytes, available, eviction_candidates_cache->size(),
			excluded_anchors.size());

	// Diagnostic-only (ARACHNE_ENABLE_TRACING build): see the
	// buildRelocationPlan_collect scope above -- this one covers only the
	// victim-selection loop, so the two phases show up as separate CSVs.
	{
	ARACHNE_TRACE_SCOPE("RegionManager", "buildRelocationPlan_evict");
	while (available + plan.immediately_reclaimable_bytes < plan.required_incremental_bytes) {
		std::size_t remaining = plan.required_incremental_bytes -
				(available + plan.immediately_reclaimable_bytes);
		std::optional<VectorId> victim = replacement_policy_->selectEvictionCandidate(
				/*excluded=*/0, remaining, *eviction_candidates_cache, excluded_anchors);
		if (!victim.has_value()) {
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: selectEvictionCandidate() returned no victim with remaining={} ({} "
					"eviction candidate(s) total, {} already excluded this pass) -- stopping victim selection short",
					batch_sequence, remaining, eviction_candidates_cache->size(), excluded_anchors.size());
			break;
		}
		auto found = std::find_if(eviction_candidates_cache->begin(), eviction_candidates_cache->end(),
				[&victim](const EvictionCandidate& candidate) { return candidate.anchor_id == *victim; });
		if (found == eviction_candidates_cache->end()) {
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: selectEvictionCandidate() returned anchor {} which isn't in the "
					"candidate list -- stopping victim selection short",
					batch_sequence, victim.value());
			break;
		}
		// Evicting *any* member of found's group without the rest would not
		// actually free found->reclaimable_bytes -- see EvictionCandidate::
		// group_members's doc comment -- so the whole named group goes into
		// plan.evictions together, in one step, not just *victim alone.
		bool is_first_group = plan.evictions.empty();
		std::vector<VectorId> group_to_evict = found->group_members;
		std::size_t evictions_size_before = plan.evictions.size();
		plan.evictions.insert(plan.evictions.end(), group_to_evict.begin(), group_to_evict.end());
		std::size_t projected = projectedReclaimableBytes(plan.evictions, /*require_unpinned=*/true);
		ARACHNE_LOG_INFO(
				"buildRelocationPlan[{}]: selected victim anchor {} and its {} group member(s) (resident_bytes={} "
				"reclaimable_bytes={} reclaimable_now_bytes={}) -- plan.evictions now {} anchor(s), projected "
				"reclaimable {} -> {}",
				batch_sequence, *victim, group_to_evict.size(), found->resident_bytes, found->reclaimable_bytes,
				found->reclaimable_now_bytes, plan.evictions.size(), plan.immediately_reclaimable_bytes, projected);
		if (coordinator_config_.max_eviction_bytes_per_pass != 0 && !is_first_group &&
				projected > coordinator_config_.max_eviction_bytes_per_pass) {
			ARACHNE_LOG_INFO(
					"buildRelocationPlan[{}]: projected {} exceeds max_eviction_bytes_per_pass={} with {} victims "
					"already selected -- dropping the just-added group and stopping",
					batch_sequence, projected, coordinator_config_.max_eviction_bytes_per_pass, plan.evictions.size());
			plan.evictions.resize(evictions_size_before);
			break;
		}
		plan.immediately_reclaimable_bytes = projected;
		// Every group member has its own EvictionCandidate entry (see
		// buildEvictionCandidates()'s doc comment) -- all of them just got
		// committed to plan.evictions together, so none should be offered
		// again as a separate, redundant selection. Grown into excluded_anchors
		// (small, O(1) inserts) instead of erased out of eviction_candidates_cache
		// (shared, large) -- see this loop's setup comment above.
		excluded_anchors.insert(group_to_evict.begin(), group_to_evict.end());
	}
	}  // end buildRelocationPlan_evict trace scope

	if (available + plan.immediately_reclaimable_bytes < plan.required_incremental_bytes) {
		ARACHNE_LOG_INFO(
				"buildRelocationPlan[{}]: giving up -- available={} + immediately_reclaimable={} still < required={} "
				"after selecting {} victim(s); retain_failed_candidates={}",
				batch_sequence, available, plan.immediately_reclaimable_bytes, plan.required_incremental_bytes,
				plan.evictions.size(), retain_failed_candidates);
		if (retain_failed_candidates) {
			std::vector<PromotionCandidate> retry;
			for (PlannedPromotion& promotion : plan.promotions) retry.push_back(std::move(promotion.candidate));
			requeueCandidates(std::move(retry));
		}
		return std::nullopt;
	}
	ARACHNE_LOG_INFO(
			"buildRelocationPlan[{}]: committed plan with {} promotion(s), {} eviction(s), "
			"required={} available={} immediately_reclaimable={}",
			batch_sequence, plan.promotions.size(), plan.evictions.size(), plan.required_incremental_bytes, available,
			plan.immediately_reclaimable_bytes);
	return plan;
}

void RegionManager::processRelocationBatch(bool retain_failed_candidates) {
	ARACHNE_TRACE_SCOPE("RegionManager", "processRelocationBatch");
	// Diagnostic-only (ARACHNE_ENABLE_TRACING build) -- see the pass-count
	// investigation this was added for (cpp/test/index/report/): coordinatorLoop()
	// passes retain_failed_candidates as exactly `!forced && !stop`, so this
	// distinguishes an ordinary, trigger_interval-paced call from one that's
	// part of a forced/stop drain's own back-to-back do-while loop (see
	// coordinatorLoop()'s own comment on that loop). Deliberately a
	// zero-duration marker, not a real timing scope -- these two scopes'
	// duration_ns is meaningless (measures only the if/else dispatch); only
	// each one's count and start_ns (still stamped at this call's true entry
	// time) matter, to bucket every processRelocationBatch call's true
	// duration (the *outer*, whole-function scope above) by which of the two
	// it belongs to and when it happened.
	if (retain_failed_candidates) {
		ARACHNE_TRACE_SCOPE("RegionManager", "processRelocationBatch_normal");
	} else {
		ARACHNE_TRACE_SCOPE("RegionManager", "processRelocationBatch_forced");
	}
	std::optional<RelocationPlan> maybe_plan =
			buildRelocationPlan(next_batch_sequence_++, retain_failed_candidates);
	if (!maybe_plan.has_value()) return;
	RelocationPlan plan = std::move(*maybe_plan);
	stat_relocation_batches_.fetch_add(1, std::memory_order_relaxed);

	const std::size_t budget = device_region_pool_->budgetBytes(gpu::MemoryKind::Data);
	const std::size_t reserved = device_region_pool_->bytesReserved(gpu::MemoryKind::Data);
	const std::size_t available = budget > reserved ? budget - reserved : 0;
	const std::size_t revalidated_reclaimable = projectedReclaimableBytes(plan.evictions, true);
	if (available + revalidated_reclaimable < plan.required_incremental_bytes) {
		ARACHNE_LOG_INFO(
				"processRelocationBatch[{}]: execution-time re-validation FAILED -- available={} + "
				"revalidated_reclaimable={} < required={} (plan had {} eviction(s) selected during planning with "
				"immediately_reclaimable={}) -- abandoning batch, retain_failed_candidates={}",
				plan.batch_sequence, available, revalidated_reclaimable, plan.required_incremental_bytes,
				plan.evictions.size(), plan.immediately_reclaimable_bytes, retain_failed_candidates);
		if (retain_failed_candidates) {
			std::vector<PromotionCandidate> retry;
			for (PlannedPromotion& promotion : plan.promotions) retry.push_back(std::move(promotion.candidate));
			requeueCandidates(std::move(retry));
		}
		return;
	}
	ARACHNE_LOG_INFO(
			"processRelocationBatch[{}]: execution-time re-validation passed (available={} revalidated_reclaimable={} "
			"required={}) -- retiring {} anchor(s), promoting {} candidate(s)",
			plan.batch_sequence, available, revalidated_reclaimable, plan.required_incremental_bytes,
			plan.evictions.size(), plan.promotions.size());

	std::vector<Region> retired = retireAnchorsNow(plan.evictions);
	ARACHNE_LOG_INFO("processRelocationBatch[{}]: retireAnchorsNow() orphaned {} region(s) out of {} evicted anchor(s)",
			plan.batch_sequence, retired.size(), plan.evictions.size());
	ReusableAllocations reusable = reclaimRegionsForPlan(
			retired, buildPromotionStorageRequests(plan.promotions));
	ARACHNE_LOG_INFO("processRelocationBatch[{}]: reclaimRegionsForPlan() reused {} of {} orphaned region(s)",
			plan.batch_sequence, reusable.size(), retired.size());

	gpu::DeviceRegionPool::TransferBatch pending;
	std::vector<PendingPromotionCommit> commits;
	std::vector<bool> retry(plan.promotions.size(), false);
	for (std::size_t i = 0; i < plan.promotions.size(); ++i) {
		PlannedPromotion& promotion = plan.promotions[i];
		std::unordered_set<RegionId> unique;
		for (RegionId region : promotion.candidate.footprint.regions) {
			if (!unique.insert(region).second) continue;
			MakeResult result = make(promotion.candidate.anchor_id, promotion.candidate.epoch,
					region, pending, commits, reusable);
			ARACHNE_LOG_INFO("processRelocationBatch[{}]: make(anchor={}, region={}) -> {}", plan.batch_sequence,
					promotion.candidate.anchor_id, region,
					result == MakeResult::Promoted        ? "Promoted"
					: result == MakeResult::NotEligible    ? "NotEligible"
					: result == MakeResult::OutOfCapacity  ? "OutOfCapacity"
																									: "Deferred");
			if (result == MakeResult::OutOfCapacity || result == MakeResult::Deferred) {
				retry[i] = true;
				break;
			}
		}
	}
	device_region_pool_->finishTransfers(pending);
	for (const auto& [region, handle] : reusable) device_region_pool_->free(handle);

	for (const PendingPromotionCommit& commit : commits) {
		bool published = false;
		{
			std::lock_guard<RegionManagerMutex> lock(mutex_);
			auto region_it = regions_.find(commit.region_id);
			if (region_it != regions_.end() &&
					region_it->second.residency_state == RegionResidencyState::Promoting &&
					region_it->second.residency_generation == commit.residency_generation &&
					currentEpochLocked(commit.anchor_id) == commit.anchor_epoch) {
				region_it->second.device = commit.device;
				region_it->second.lease = commit.lease;
				region_it->second.residency_state = RegionResidencyState::Resident;
				dependents_[commit.region_id].insert(commit.anchor_id);
				dependencies_[commit.anchor_id].insert(commit.region_id);
				published = true;
			} else if (region_it != regions_.end() &&
							 region_it->second.residency_state == RegionResidencyState::Promoting &&
							 region_it->second.residency_generation == commit.residency_generation) {
				region_it->second.residency_state = RegionResidencyState::HostOnly;
			}
		}
		ARACHNE_LOG_INFO("processRelocationBatch[{}]: publish region={} anchor={} generation={} -> {}",
				plan.batch_sequence, commit.region_id, commit.anchor_id, commit.residency_generation,
				published ? "Resident" : "publish failed (stale generation/epoch), reverted to HostOnly");
		if (published) {
			stat_regions_promoted_.fetch_add(1, std::memory_order_relaxed);
		} else {
			if (IRegion* target = adapter_->resolveRegion(commit.region_id)) target->releaseWriteLease(commit.lease);
			device_region_pool_->free(commit.device);
		}
	}

	std::vector<PromotionCandidate> retry_candidates;
	for (std::size_t i = 0; i < plan.promotions.size(); ++i) {
		PlannedPromotion& promotion = plan.promotions[i];
		std::vector<RegionId> resident_regions = regionsOf(promotion.candidate.anchor_id);
		bool resident = !resident_regions.empty();
		if (resident) {
			replacement_policy_->onPromotionCommitted(promotion.candidate.anchor_id, promotion.admission);
			if (routing_cache_ != nullptr) {
				routing_cache_->ensure(promotion.candidate.anchor_id, promotion.candidate.vectorView(),
						kDefaultAnchorMaxDistance);
			}
			// Group assignment uses this Anchor's final, actually-committed
			// footprint (not the originally-requested one -- some Regions in
			// the original footprint may have come back NotEligible/Deferred
			// above and simply aren't part of resident_regions).
			assignAnchorToGroup(promotion.candidate.anchor_id, resident_regions);
		}
		if (retry[i] && retain_failed_candidates) {
			retry_candidates.push_back(std::move(promotion.candidate));
		}
	}
	if (!retry_candidates.empty()) requeueCandidates(std::move(retry_candidates));
}

void RegionManager::evictAnchorNow(VectorId anchor_id) {
	ARACHNE_TRACE_SCOPE("RegionManager", "evictAnchorNow");
	reclaimRegions(retireAnchorsNow({anchor_id}));
}

std::vector<Region> RegionManager::retireAnchorsNow(const std::vector<VectorId>& anchor_ids) {
	std::vector<Region> snapshots;
	for (VectorId anchor_id : anchor_ids) {
		std::vector<RegionId> orphaned = forget(anchor_id);
		replacement_policy_->onAnchorEvicted(anchor_id);
		if (routing_cache_ != nullptr) routing_cache_->erase(anchor_id);
		{
			std::lock_guard<RegionManagerMutex> lock(mutex_);
			for (RegionId region_id : orphaned) {
				auto it = regions_.find(region_id);
				if (it == regions_.end() || it->second.residency_state != RegionResidencyState::Resident) continue;
				it->second.residency_state = RegionResidencyState::Retiring;
				++it->second.residency_generation;
				snapshots.push_back(it->second);
			}
		}
	}
	stat_anchor_evictions_.fetch_add(anchor_ids.size(), std::memory_order_relaxed);
	return snapshots;
}

RegionManager::ReusableAllocations RegionManager::reclaimRegionsForPlan(
		const std::vector<Region>& snapshots,
		const std::vector<PromotionStorageRequest>& requests) {
	std::vector<Region> ready;
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (const Region& snapshot : snapshots) {
			auto it = regions_.find(snapshot.id);
			if (it == regions_.end() || it->second.residency_state != RegionResidencyState::Retiring ||
					it->second.residency_generation != snapshot.residency_generation) continue;
			if (it->second.residency_pins != 0) {
				pending_reclaims_.push_back(it->second);
				continue;
			}
			if (it->second.device.valid()) ready.push_back(it->second);
		}
	}
	ReusableAllocations reused;
	if (ready.empty()) return reused;

	writeBackDirtyRegions(ready);
	std::unordered_set<std::uint64_t> reused_handles;
	std::vector<PromotionStorageRequest> ordered_requests = requests;
	std::sort(ordered_requests.begin(), ordered_requests.end(),
			[](const PromotionStorageRequest& a, const PromotionStorageRequest& b) {
				return a.reserved_bytes > b.reserved_bytes;
			});
	for (const PromotionStorageRequest& request : ordered_requests) {
		Region* best = nullptr;
		std::size_t best_capacity = std::numeric_limits<std::size_t>::max();
		for (Region& slot : ready) {
			if (reused_handles.contains(slot.device.id)) continue;
			std::size_t capacity = reservedRegionBytes(slot);
			const bool sufficiently_utilized =
					request.reserved_bytes >= MinimumUtilizedBytes(
							capacity, coordinator_config_.near_fit_min_utilization_percent);
			if (capacity >= request.reserved_bytes && sufficiently_utilized && capacity < best_capacity) {
				best = &slot;
				best_capacity = capacity;
			}
		}
		if (best != nullptr && device_region_pool_->tryReuse(best->device, request.logical_bytes)) {
			reused.emplace(request.region_id, best->device);
			reused_handles.insert(best->device.id);
			stat_near_fit_reuses_.fetch_add(1, std::memory_order_relaxed);
		}
	}

	for (const Region& region : ready) {
		if (!reused_handles.contains(region.device.id)) device_region_pool_->free(region.device);
		if (IRegion* target = adapter_->resolveRegion(region.id)) target->releaseWriteLease(region.lease);
	}
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (const Region& region : ready) {
			auto it = regions_.find(region.id);
			if (it == regions_.end() || it->second.residency_state != RegionResidencyState::Retiring ||
					it->second.residency_generation != region.residency_generation) continue;
			it->second.lease = LeaseHandle{};
			it->second.device = gpu::DeviceRegionHandle{};
			it->second.residency_state = RegionResidencyState::HostOnly;
		}
	}
	stat_regions_evicted_.fetch_add(ready.size(), std::memory_order_relaxed);
	return reused;
}

void RegionManager::reclaimRegions(const std::vector<Region>& snapshots) {
	ARACHNE_TRACE_SCOPE("RegionManager", "reclaimRegions");
	std::vector<Region> resident;
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (const Region& snapshot : snapshots) {
			auto it = regions_.find(snapshot.id);
			if (it == regions_.end() ||
					it->second.residency_state != RegionResidencyState::Retiring ||
					it->second.residency_generation != snapshot.residency_generation) {
				continue;
			}
			if (it->second.residency_pins != 0) {
				pending_reclaims_.push_back(it->second);
				continue;
			}
			if (it->second.device.valid()) resident.push_back(it->second);
		}
	}
	if (resident.empty()) return;

	writeBackDirtyRegions(resident);  // one batched gather across every Region here, not one per Region
	for (const Region& region : resident) device_region_pool_->free(region.device);
	for (const Region& region : resident) {
		if (IRegion* target = adapter_->resolveRegion(region.id)) target->releaseWriteLease(region.lease);
	}
	{
		std::lock_guard<RegionManagerMutex> lock(mutex_);
		for (const Region& region : resident) {
			auto it = regions_.find(region.id);
			if (it == regions_.end() ||
					it->second.residency_state != RegionResidencyState::Retiring ||
					it->second.residency_generation != region.residency_generation) {
				continue;
			}
			it->second.lease = LeaseHandle{};
			it->second.device = gpu::DeviceRegionHandle{};
			it->second.residency_state = RegionResidencyState::HostOnly;
		}
	}
	stat_regions_evicted_.fetch_add(resident.size(), std::memory_order_relaxed);
}

namespace {
bool AnyDirtyWordSet(const std::vector<std::byte>& header) {
	const auto* words = reinterpret_cast<const std::uint64_t*>(header.data());
	std::size_t word_count = header.size() / gpu::kDirtyWordBytes;
	for (std::size_t i = 0; i < word_count; ++i) {
		if (words[i] != 0) return true;
	}
	return false;
}
}  // namespace

void RegionManager::writeBackDirtyRegions(const std::vector<Region>& regions) {
	ARACHNE_TRACE_SCOPE("RegionManager", "writeBackDirtyRegions");
	const std::size_t n = regions.size();
	std::vector<std::size_t> header_bytes(n);
	std::vector<std::vector<std::byte>> header_buffers(n);
	std::vector<bool> dirty(n, true);  // conservative default for Regions with no header -- see doc comment

	{
		std::vector<gpu::DeviceRegionPool::Lease> pending;
		for (std::size_t i = 0; i < n; ++i) {
			header_bytes[i] = gpu::DirtyHeaderBytes(regions[i].host.bytes, regions[i].host.subregion_bytes);
			if (header_bytes[i] == 0) continue;  // no header to gather for this Region
			header_buffers[i].resize(header_bytes[i]);
			device_region_pool_->enqueueCopyToHost(regions[i].device, header_buffers[i].data(), header_bytes[i],
																							/*src_offset=*/0, pending);
		}
		device_region_pool_->flush();
	}

	for (std::size_t i = 0; i < n; ++i) {
		if (header_bytes[i] > 0) dirty[i] = AnyDirtyWordSet(header_buffers[i]);
	}

	std::size_t written_back = 0;
	{
		std::vector<gpu::DeviceRegionPool::Lease> pending;
		for (std::size_t i = 0; i < n; ++i) {
			if (!dirty[i] || regions[i].host.ptr == nullptr || regions[i].host.bytes == 0) {
				ARACHNE_LOG_DEBUG("writeBackDirtyRegions: region {} is clean, skipping write-back", regions[i].id);
				continue;
			}
			device_region_pool_->enqueueCopyToHost(regions[i].device, regions[i].host.ptr, regions[i].host.bytes,
																							header_bytes[i], pending);
			++written_back;
		}
		device_region_pool_->flush();
	}

	stat_regions_written_back_.fetch_add(written_back, std::memory_order_relaxed);
	ARACHNE_LOG_DEBUG("writeBackDirtyRegions: wrote back {} of {} region(s)", written_back, n);
}

}  // namespace arachne
