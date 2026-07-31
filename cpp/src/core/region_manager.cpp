#include "core/region_manager.hpp"

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
}  // namespace

RegionManager::RegionManager(std::unique_ptr<ReplacementPolicy> replacement_policy)
		: replacement_policy_(std::move(replacement_policy)) {
	if (replacement_policy_ == nullptr) replacement_policy_ = std::make_unique<FifoReplacementPolicy>();
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
	if (vector.data != nullptr && vector.dim > 0) {
		const auto* bytes = static_cast<const std::byte*>(vector.data);
		candidate.vector_bytes.assign(bytes, bytes + static_cast<std::size_t>(vector.dim) * VectorElementSize(vector.dtype));
	}

	std::lock_guard<RegionManagerMutex> lock(mutex_);
	candidate.epoch = currentEpochLocked(anchor_id);
	pending_promotions_.push_back(std::move(candidate));
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

	std::vector<Region> snapshots;
	snapshots.reserve(orphaned.size());
	for (RegionId region_id : orphaned) {
		Region region = regionOf(region_id);  // snapshot BEFORE clearResidency() below overwrites the live record
		if (adapter_ != nullptr) {
			if (IRegion* target = adapter_->resolveRegion(region_id)) target->releaseWriteLease(region.lease);
		}
		clearResidency(region_id);
		snapshots.push_back(region);
	}

	stat_anchor_evictions_.fetch_add(1, std::memory_order_relaxed);

	std::lock_guard<RegionManagerMutex> lock(mutex_);
	for (Region& region : snapshots) pending_reclaims_.push_back(std::move(region));
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
	std::lock_guard<RegionManagerMutex> lock(mutex_);
	if (device_region_pool_ != nullptr) result.gpu_bytes_allocated = device_region_pool_->bytesAllocated();
	return result;
}

void RegionManager::coordinatorLoop() {
	while (true) {
		std::unique_lock<RegionManagerMutex> lock(mutex_);
		coordinator_cv_.wait_for(lock, coordinator_config_.trigger_interval,
															[this] { return coordinator_stop_requested_ || coordinator_force_wake_; });
		bool forced = coordinator_force_wake_;
		coordinator_force_wake_ = false;
		bool stop = coordinator_stop_requested_;

		coordinator_busy_ = true;
		std::vector<PromotionCandidate> admitted(std::make_move_iterator(pending_promotions_.begin()),
																							std::make_move_iterator(pending_promotions_.end()));
		pending_promotions_.clear();
		std::vector<Region> reclaims(std::make_move_iterator(pending_reclaims_.begin()),
																 std::make_move_iterator(pending_reclaims_.end()));
		pending_reclaims_.clear();
		lock.unlock();

		// Ingestion happens every wakeup regardless of what runs below -- this
		// keeps the policy's view current without waiting for a trigger.
		for (PromotionCandidate& candidate : admitted) {
			replacement_policy_->enqueueCandidate(std::move(candidate));
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
		if (forced || stop || replacement_policy_->onRelocationTrigger()) {
			processPromotions();
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
		std::size_t bytes, std::vector<gpu::DeviceRegionPool::Lease>& pending) {
	std::optional<gpu::DeviceRegionHandle> handle = device_region_pool_->tryAllocate(bytes);
	if (handle.has_value()) return handle;

	if (!device_region_pool_->hasCapacity(bytes)) {
		return std::nullopt;  // genuinely over budget -- compacting frees nothing new
	}

	// Flush and release any Lease already pinned earlier in this batch first:
	// compact() only relocates unpinned allocations, so this widens its pool
	// of movable blocks instead of excluding Regions this same batch pinned.
	if (!pending.empty()) {
		device_region_pool_->flush();
		pending.clear();
	}
	device_region_pool_->compact(gpu::MemoryKind::Data, bytes);
	stat_compactions_total_.fetch_add(1, std::memory_order_relaxed);
	return device_region_pool_->tryAllocate(bytes);
}

RegionManager::MakeResult RegionManager::make(VectorId anchor_id, RegionId region,
																							 std::vector<gpu::DeviceRegionPool::Lease>& pending,
																							 std::vector<std::vector<std::byte>>& zero_headers) {
	ARACHNE_TRACE_SCOPE("RegionManager", "make");
	for (RegionId existing : regionsOf(anchor_id)) {
		if (existing == region) return MakeResult::Promoted;  // already a dependent
	}

	if (!isRegistered(region)) {
		ARACHNE_LOG_DEBUG("make: region {} is not registered, anchor {} cannot depend on it", region, anchor_id);
		return MakeResult::NotEligible;
	}

	Region snapshot = regionOf(region);
	if (!snapshot.lease.valid()) {
		IRegion* target = adapter_->resolveRegion(region);
		if (target == nullptr) return MakeResult::NotEligible;

		LeaseHandle lease = target->acquireWriteLease();
		if (!lease.valid()) {
			ARACHNE_LOG_DEBUG("make: region {} not lease-eligible for anchor {}", region, anchor_id);
			return MakeResult::NotEligible;
		}

		std::size_t header_bytes = gpu::DirtyHeaderBytes(snapshot.host.bytes, snapshot.host.subregion_bytes);
		std::optional<gpu::DeviceRegionHandle> device =
				allocateWithCompaction(header_bytes + snapshot.host.bytes, pending);
		if (!device.has_value()) {
			target->releaseWriteLease(lease);
			ARACHNE_LOG_DEBUG("make: region {} ({} bytes) out of GPU capacity for anchor {}", region,
												 header_bytes + snapshot.host.bytes, anchor_id);
			return MakeResult::OutOfCapacity;
		}
		if (header_bytes > 0) {
			zero_headers.emplace_back(header_bytes, std::byte{0});
			device_region_pool_->enqueueCopyFromHost(*device, zero_headers.back().data(), header_bytes,
																								/*dst_offset=*/0, pending);
		}
		device_region_pool_->enqueueCopyFromHost(*device, snapshot.host.ptr, snapshot.host.bytes, header_bytes,
																							pending);
		setDevice(region, *device);
		setLease(region, lease);
		stat_regions_promoted_.fetch_add(1, std::memory_order_relaxed);
	}

	// No replacement_policy_ notification here -- selectNextPromotionCandidate()
	// already recorded anchor_id for eviction-ordering purposes before make()
	// was ever called for it.
	addDependency(anchor_id, region);

	ARACHNE_LOG_DEBUG("make: anchor {} now depends on region {}", anchor_id, region);
	return MakeResult::Promoted;
}

void RegionManager::processPromotions() {
	ARACHNE_TRACE_SCOPE("RegionManager", "processPromotions");
	std::vector<gpu::DeviceRegionPool::Lease> pending;
	std::vector<std::vector<std::byte>> zero_headers;

	while (std::optional<PromotionCandidate> candidate = replacement_policy_->selectNextPromotionCandidate()) {
		{
			std::lock_guard<RegionManagerMutex> lock(mutex_);
			if (currentEpochLocked(candidate->anchor_id) != candidate->epoch) {
				ARACHNE_LOG_DEBUG("processPromotions: anchor {} epoch stale, discarding candidate", candidate->anchor_id);
				continue;
			}
		}

		bool any_promoted = false;
		for (RegionId region : candidate->footprint.regions) {
			MakeResult result = make(candidate->anchor_id, region, pending, zero_headers);
			if (result == MakeResult::Promoted) {
				any_promoted = true;
				continue;
			}
			if (result == MakeResult::NotEligible) {
				ARACHNE_LOG_DEBUG("processPromotions: region {} not eligible for anchor {}, skipping", region,
													 candidate->anchor_id);
				continue;
			}

			while (result == MakeResult::OutOfCapacity) {
				std::optional<VectorId> victim = replacement_policy_->selectNextEvictionCandidate(candidate->anchor_id);
				if (!victim.has_value()) {
					ARACHNE_LOG_DEBUG("processPromotions: no eviction candidate left, anchor {} region {} not promoted",
														 candidate->anchor_id, region);
					break;
				}

				// Flush and release every Lease from this pass before evicting:
				// `victim` may depend on a Region this pass already promoted and
				// is still pinning via `pending`. Without this,
				// evictAnchorNow()'s free() would await that same Lease -- a real
				// deadlock (the Coordinator waiting on itself), not just a missed
				// optimization. Safe unconditionally: flush() only proves
				// already-enqueued copies landed, it doesn't invalidate `pending`.
				if (!pending.empty()) {
					device_region_pool_->flush();
					pending.clear();
				}
				evictAnchorNow(*victim);
				result = make(candidate->anchor_id, region, pending, zero_headers);
			}

			if (result == MakeResult::Promoted) {
				any_promoted = true;
			} else {
				ARACHNE_LOG_DEBUG("processPromotions: region {} still unavailable for anchor {} (result={})", region,
													 candidate->anchor_id, static_cast<int>(result));
			}
		}

		// Register in RoutingCache only if at least one Region in this
		// candidate's footprint actually became a dependency -- a candidate
		// that went entirely NotEligible/OutOfCapacity gained no GPU residency
		// for a future query to be routed to.
		if (any_promoted && routing_cache_ != nullptr) {
			routing_cache_->ensure(candidate->anchor_id, candidate->vectorView(), kDefaultAnchorMaxDistance);
		}
	}

	// One batched flush for the whole pass, not one per Anchor -- mirrors
	// reclaimRegions()/writeBackDirtyRegions()'s batched-gather shape for the
	// opposite (device-to-host) direction.
	device_region_pool_->flush();
}

void RegionManager::evictAnchorNow(VectorId anchor_id) {
	ARACHNE_TRACE_SCOPE("RegionManager", "evictAnchorNow");
	std::vector<RegionId> orphaned = forget(anchor_id);
	replacement_policy_->onAnchorEvicted(anchor_id);
	if (routing_cache_ != nullptr) routing_cache_->erase(anchor_id);

	if (orphaned.empty()) return;

	std::vector<Region> snapshots;
	snapshots.reserve(orphaned.size());
	for (RegionId region_id : orphaned) {
		Region region = regionOf(region_id);  // still registered; residency not yet cleared
		if (IRegion* target = adapter_->resolveRegion(region_id)) {
			target->releaseWriteLease(region.lease);
		}
		snapshots.push_back(region);
	}

	reclaimRegions(snapshots);

	for (RegionId region_id : orphaned) clearResidency(region_id);
	stat_anchor_evictions_.fetch_add(1, std::memory_order_relaxed);

	ARACHNE_LOG_DEBUG("evictAnchorNow: released anchor {}'s dependency on {} region(s)", anchor_id, orphaned.size());
}

void RegionManager::reclaimRegions(const std::vector<Region>& snapshots) {
	ARACHNE_TRACE_SCOPE("RegionManager", "reclaimRegions");
	std::vector<Region> resident;
	for (const Region& region : snapshots) {
		if (region.device.valid()) resident.push_back(region);
	}
	if (resident.empty()) return;

	writeBackDirtyRegions(resident);  // one batched gather across every Region here, not one per Region
	for (const Region& region : resident) device_region_pool_->free(region.device);
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
