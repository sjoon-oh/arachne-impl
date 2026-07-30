#include "core/region_manager.hpp"

#include <stdexcept>
#include <utility>

#include "gpu/dirty_header.hpp"
#include "logging.hpp"

namespace arachne {

namespace {
// Placeholder policy value for the radius a newly registered Anchor is
// given in RoutingCache -- relocated from Controller (see
// commitSearch()/commitInsert()'s old doc comments) now that RegionManager
// is the one calling RoutingCache::ensure(). RoutingCache itself is
// radius-agnostic (each Anchor carries its own max_distance); a real
// per-query threshold (e.g. derived from query density/confidence) is
// future work -- this constant stands in for it.
constexpr float kDefaultAnchorMaxDistance = 1e-3f;
}  // namespace

RegionManager::RegionManager(std::unique_ptr<ReplacementPolicy> replacement_policy)
		: replacement_policy_(std::move(replacement_policy)) {
	if (replacement_policy_ == nullptr) replacement_policy_ = std::make_unique<FifoReplacementPolicy>();
}

RegionManager::~RegionManager() { shutdown(); }

void RegionManager::registerRegion(RegionId id, HostRegionView host) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (regions_.find(id) != regions_.end()) return;

	Region region;
	region.id = id;
	region.host = host;
	regions_.emplace(id, region);
}

bool RegionManager::isRegistered(RegionId id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	return regions_.find(id) != regions_.end();
}

Region RegionManager::regionOf(RegionId id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it == regions_.end()) {
		throw std::invalid_argument("RegionManager: region is not registered");
	}
	return it->second;
}

std::vector<RegionId> RegionManager::regionsOf(VectorId anchor_id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = dependencies_.find(anchor_id);
	if (it == dependencies_.end()) return {};
	return std::vector<RegionId>(it->second.begin(), it->second.end());
}

bool RegionManager::addDependency(VectorId anchor_id, RegionId region_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (regions_.find(region_id) == regions_.end()) return false;

	dependents_[region_id].insert(anchor_id);
	dependencies_[anchor_id].insert(region_id);
	return true;
}

bool RegionManager::removeDependency(VectorId anchor_id, RegionId region_id) {
	std::lock_guard<std::mutex> lock(mutex_);
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
	std::lock_guard<std::mutex> lock(mutex_);
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
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it != regions_.end()) it->second.lease = lease;
}

void RegionManager::setDevice(RegionId id, gpu::DeviceRegionHandle device) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it != regions_.end()) it->second.device = device;
}

void RegionManager::clearResidency(RegionId id) {
	std::lock_guard<std::mutex> lock(mutex_);
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
	std::lock_guard<std::mutex> lock(mutex_);
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
		std::lock_guard<std::mutex> lock(mutex_);
		if (!coordinator_running_) return;
		coordinator_stop_requested_ = true;
		coordinator_force_wake_ = true;
	}
	coordinator_cv_.notify_all();

	if (coordinator_.joinable()) coordinator_.join();

	std::lock_guard<std::mutex> lock(mutex_);
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
	// A pure enqueue onto RegionManager's own MPSC intake queue -- no
	// replacement_policy_ interaction here at all. Only the Coordinator
	// thread (the queue's single consumer) ever hands candidates to the
	// policy -- see the class doc comment (region_manager.hpp) and
	// ReplacementPolicy::enqueueCandidate()'s own doc comment for why.
	PromotionCandidate candidate;
	candidate.anchor_id = anchor_id;
	candidate.footprint = std::move(footprint);
	candidate.vector_dim = vector.dim;
	candidate.vector_dtype = vector.dtype;
	if (vector.data != nullptr && vector.dim > 0) {
		const auto* bytes = static_cast<const std::byte*>(vector.data);
		candidate.vector_bytes.assign(bytes, bytes + static_cast<std::size_t>(vector.dim) * VectorElementSize(vector.dtype));
	}

	std::lock_guard<std::mutex> lock(mutex_);
	candidate.epoch = currentEpochLocked(anchor_id);
	pending_promotions_.push_back(std::move(candidate));
}

void RegionManager::releaseAnchor(VectorId anchor_id) {
	// Immediate: dependency-graph bookkeeping, write-lease release, and
	// clearResidency() all happen synchronously here -- so a Region orphaned
	// by `anchor_id` is instantly eligible to be re-promoted fresh for a
	// different Anchor (make() will correctly see an invalid lease and start
	// over) without that new promotion racing or colliding with this
	// Region's still-pending GPU reclaim below, which is captured into an
	// independent snapshot and never looked up live again.
	std::vector<RegionId> orphaned = forget(anchor_id);

	// replacement_policy_ must stop tracking anchor_id here regardless of
	// whether forget() actually freed a Region -- see the historical bug this
	// guards against: an Anchor evicted from a still-multiply-depended-on
	// Region must still stop being FIFO-selectable, or a capacity-retry loop
	// elsewhere could re-select it forever without making progress.
	replacement_policy_->onAnchorEvicted(anchor_id);

	// Bump the epoch (see PromotionCandidate's own doc comment) *before*
	// erasing from RoutingCache below -- any PromotionCandidate for
	// anchor_id already sitting in pending_promotions_/the policy's own
	// storage, enqueued before this call, now carries a stale epoch and will
	// be discarded by processPromotions() rather than acted on.
	{
		std::lock_guard<std::mutex> lock(mutex_);
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

	std::lock_guard<std::mutex> lock(mutex_);
	for (Region& region : snapshots) pending_reclaims_.push_back(std::move(region));
}

void RegionManager::recordTraversal(const RegionFootprint& touched) {
	std::unordered_set<VectorId> anchors;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		for (RegionId region_id : touched.regions) {
			auto it = dependents_.find(region_id);
			if (it == dependents_.end()) continue;  // unregistered, or currently orphaned
			anchors.insert(it->second.begin(), it->second.end());
		}
	}
	for (VectorId anchor_id : anchors) replacement_policy_->onAnchorTouched(anchor_id);
}

void RegionManager::waitIdle() {
	std::unique_lock<std::mutex> lock(mutex_);
	coordinator_force_wake_ = true;
	lock.unlock();
	coordinator_cv_.notify_all();

	lock.lock();
	idle_cv_.wait(lock, [this] {
		// pending_promotions_/pending_reclaims_ being empty only proves
		// RegionManager's own intake queue has been drained *into* the policy
		// -- it does not prove the policy has actually offered every admitted
		// candidate back out via selectNextPromotionCandidate() yet (it may be
		// holding some, or onRelocationTrigger() may have been declining before
		// force_wake_ overrode it) -- so hasPendingCandidates() must also be
		// checked, or waitIdle() could return before every requested promotion
		// has actually been attempted.
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
	std::lock_guard<std::mutex> lock(mutex_);
	if (device_region_pool_ != nullptr) result.gpu_bytes_allocated = device_region_pool_->bytesAllocated();
	return result;
}

void RegionManager::coordinatorLoop() {
	while (true) {
		std::unique_lock<std::mutex> lock(mutex_);
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

		// Ingestion: hand every freshly-drained candidate to the replacement
		// policy's own storage -- see ReplacementPolicy::enqueueCandidate()'s
		// doc comment. Happens every wakeup, independent of whether execution
		// runs below -- this is the "background work" that keeps the policy's
		// view current without waiting for a trigger.
		for (PromotionCandidate& candidate : admitted) {
			replacement_policy_->enqueueCandidate(std::move(candidate));
		}

		// Reclaims are unrelated to the replacement policy (already-decided
		// cleanup work from releaseAnchor()/evictAnchorNow(), not a fresh
		// decision) -- process them unconditionally, same as before. Frees up
		// capacity a promotion in this same pass might otherwise have needed to
		// evict something else for.
		if (!reclaims.empty()) reclaimRegions(reclaims);

		// Execution: gated by the policy's own onRelocationTrigger() -- unless
		// force-woken or stopping, either of which must guarantee every
		// admitted candidate has actually been offered, bypassing whatever
		// timing preference the policy would otherwise apply. `stop` (not just
		// `forced`) is checked here too: `forced` is a one-shot flag consumed
		// on the very next wakeup, but coordinator_stop_requested_ stays true
		// across every remaining iteration until this loop actually exits, so
		// shutdown() keeps forcing execution on every iteration -- not just the
		// first -- until the policy's own backlog is genuinely drained (see the
		// break condition below), rather than risking the loop exiting early or
		// spinning on a policy that keeps declining.
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

	// Same hazard as processPromotions()'s eviction-retry loop: compact()
	// touches every live Data-kind allocation, including Regions already
	// promoted earlier in this same Coordinator pass and still holding an
	// open Lease in `pending` (make()'s caller only flushes/releases it at
	// the very end of the whole batch) -- flush and release those first, or
	// compact()'s own awaitQuiescentLocked() would deadlock waiting on a
	// Lease this same call is still holding open.
	if (!pending.empty()) {
		device_region_pool_->flush();
		pending.clear();
	}
	device_region_pool_->compact(gpu::MemoryKind::Data);
	stat_compactions_total_.fetch_add(1, std::memory_order_relaxed);
	return device_region_pool_->tryAllocate(bytes);
}

RegionManager::MakeResult RegionManager::make(VectorId anchor_id, RegionId region,
																							 std::vector<gpu::DeviceRegionPool::Lease>& pending,
																							 std::vector<std::vector<std::byte>>& zero_headers) {
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
	// already recorded `anchor_id` for eviction-ordering purposes, at the
	// moment it offered this candidate, before make() was ever called for it
	// (see that method's own doc comment).
	addDependency(anchor_id, region);

	ARACHNE_LOG_DEBUG("make: anchor {} now depends on region {}", anchor_id, region);
	return MakeResult::Promoted;
}

void RegionManager::processPromotions() {
	std::vector<gpu::DeviceRegionPool::Lease> pending;
	std::vector<std::vector<std::byte>> zero_headers;

	while (std::optional<PromotionCandidate> candidate = replacement_policy_->selectNextPromotionCandidate()) {
		{
			std::lock_guard<std::mutex> lock(mutex_);
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

				// Flush and release every Lease accumulated so far in this pass
				// *before* evicting -- a Region already promoted earlier in this
				// same batch is still holding an open Lease in `pending` (it isn't
				// released until `pending` itself is destroyed, at the very end of
				// this function), and `victim` may turn out to depend on exactly
				// that Region. Without this, evictAnchorNow()'s free() would
				// awaitQuiescentLocked() on a Lease this same function is still
				// holding open -- a real deadlock (the Coordinator thread waiting
				// on itself), not just a missed optimization. Safe to flush before
				// we know it's needed: flush() only proves already-enqueued copies
				// landed, it doesn't invalidate anything `pending` still holds.
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

		// Register the Anchor in RoutingCache now that its residency actually
		// changed -- see the class doc comment (region_manager.hpp) for why
		// this replaces Controller's old, unconditional commitSearch()/
		// commitInsert() calls. Only if at least one Region in this
		// candidate's footprint actually became a dependency: a candidate that
		// went entirely NotEligible/OutOfCapacity gained no GPU residency at
		// all, so there is nothing here for a future query to be routed to.
		if (any_promoted && routing_cache_ != nullptr) {
			routing_cache_->ensure(candidate->anchor_id, candidate->vectorView(), kDefaultAnchorMaxDistance);
		}
	}

	// One batched flush for every host-to-device copy enqueued above across
	// the *whole pass* -- not just one Anchor's footprint -- one
	// scatter-gather round trip for everything the policy offered this
	// trigger, mirroring reclaimRegions()/writeBackDirtyRegions()'s own
	// batched-gather shape for the opposite (device-to-host) direction.
	device_region_pool_->flush();
}

void RegionManager::evictAnchorNow(VectorId anchor_id) {
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
