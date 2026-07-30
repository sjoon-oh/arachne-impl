#include "core/controller.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "logging.hpp"

namespace arachne {

namespace {
constexpr std::uint64_t kDriftWindowSize = 128;

// Placeholder policy value for the radius a newly registered Anchor is
// given in the RoutingCache. RoutingCache itself is radius-agnostic (each
// Anchor carries its own max_distance); Core is where that radius policy
// belongs, but a real per-query threshold (e.g. derived from query
// density/confidence) is future work -- this constant stands in for it.
constexpr float kDefaultAnchorMaxDistance = 1e-3f;

// Placeholder: how many candidate neighbors the lookup traversal insert()
// runs first (see its doc comment) asks for. Core never interprets this
// beyond passing it through -- what actually matters for placement quality
// (HNSW's efConstruction or similar) is an adapter/index tuning concern the
// returned TraverseResult::hint is free to reflect however it needs to,
// regardless of this nominal top_k.
constexpr std::uint32_t kInsertionLookupTopK = 1;
}  // namespace

Controller::Controller(IndexAdapter& adapter, RoutingCache& routing_cache, SchedulingConfig scheduling_config,
												std::unique_ptr<ReplacementPolicy> replacement_policy,
												std::size_t gpu_data_budget_bytes, std::size_t gpu_metadata_budget_bytes)
	: adapter_(adapter),
		routing_cache_(routing_cache),
		scheduler_(scheduling_config),
		replacement_policy_(std::move(replacement_policy)),
		device_(/*device_id=*/0, gpu::AllocationPolicy::Pooled, gpu_data_budget_bytes, gpu_metadata_budget_bytes),
		device_region_pool_(device_) {
	if (replacement_policy_ == nullptr) replacement_policy_ = std::make_unique<FifoReplacementPolicy>();
	scheduler_.start(adapter_);
}

SearchResult Controller::search(const Query& query) {
	SearchPlan plan = routeSearch(query);
	TraverseResult result = dispatch(plan.primary);
	bool final_was_hybrid = (plan.primary.mode == ExecutionMode::Hybrid);

	if (plan.fallback_to_hybrid && !result.completed_within_scope) {
		TraverseRequest fallback_request{query, ExecutionMode::Hybrid, {}};
		result = dispatch(fallback_request);
		final_was_hybrid = true;
	}

	return commitSearch(plan, result, final_was_hybrid);
}

InsertResult Controller::insert(const Record& record) {
	// Claim record.id before doing anything else -- see insert()'s doc
	// comment (controller.hpp). The insert-into-set-and-check-result-atomically
	// pattern here is what makes two concurrent insert() calls for the same
	// id race safely: exactly one of them observes `.second == true` and
	// proceeds, the other sees false and bails out immediately.
	{
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		if (!live_ids_.insert(record.id).second) {
			ARACHNE_LOG_WARN("insert: id {} is already live, rejecting duplicate insert", record.id);
			return InsertResult{false};
		}
	}

	try {
		// Step 1 (Traversal): find where this new vector belongs -- candidate
		// neighbors to link to, a cluster to join, or whatever else a concrete
		// index's algorithm needs (see TraverseResult::hint) -- using the same
		// anchor-routing decision search() itself uses, so a repeatedly-inserted-
		// near Anchor gets GpuOnly lookups the same way a repeatedly-queried one
		// does.
		Query lookup_query{record.vector, kInsertionLookupTopK};
		RoutingDecision decision = route(lookup_query);
		TraverseRequest lookup{lookup_query, decision.gpu_only ? ExecutionMode::GpuOnly : ExecutionMode::Hybrid,
													 decision.predicted_scope};
		TraverseResult candidates = dispatch(lookup);

		// Step 2 (Modification): apply the insert using what the traversal found.
		InsertPlan plan = routeInsert(record, std::move(candidates));
		ModifyResult result = dispatch(plan.request);
		InsertResult final_result = commitInsert(plan, result);

		if (!final_result.ok) {
			// Never actually landed -- free the id back up rather than leaving
			// it permanently unusable.
			std::lock_guard<std::mutex> lock(live_ids_mutex_);
			live_ids_.erase(record.id);
		}
		return final_result;
	} catch (...) {
		// dispatch() can throw (e.g. a GpuOnly request reaching an adapter's
		// unimplemented traverseDevice()/modifyDevice(), see IndexAdapter's doc
		// comment) -- record.id was never actually inserted either way, so it
		// must not stay claimed.
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		live_ids_.erase(record.id);
		throw;
	}
}

DeleteResult Controller::remove(VectorId id) {
	RemovePlan plan = routeRemove(id);
	ModifyResult result = dispatch(plan.request);
	DeleteResult final_result = commitRemove(plan, result);

	if (final_result.ok) {
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		live_ids_.erase(id);
	}
	return final_result;
}

Controller::SearchPlan Controller::routeSearch(const Query& query) {
	RoutingDecision decision = route(query);

	SearchPlan plan;
	plan.primary.query = query;
	plan.primary.mode = ExecutionMode::Hybrid;
	plan.primary.scope = {};
	plan.fallback_to_hybrid = false;
	if (decision.gpu_only) {
		plan.primary.mode = ExecutionMode::GpuOnly;
		// `decision` is never read again below except for eligibility.
		plan.primary.scope = decision.predicted_scope;
		plan.fallback_to_hybrid = true;
	}
	return plan;
}

Controller::InsertPlan Controller::routeInsert(const Record& record, TraverseResult candidates) {
	InsertPlan plan;
	plan.anchor_id = record.id;

	plan.request.op = ModifyOp::Insert;
	plan.request.record = record;
	plan.request.mode = ExecutionMode::Hybrid;
	// `candidates` (the lookup traversal's result) isn't read again after
	// this function, so its hint is moved rather than copied -- see
	// OpaqueData's doc comment for why Core carries it without interpreting
	// it, and routeInsert()'s own doc comment (controller.hpp) for why this
	// takes `candidates` by value in the first place.
	plan.request.scope = candidates.touched;
	plan.request.hint = std::move(candidates.hint);

	for (RegionId region_id : region_manager_.regionsOf(plan.anchor_id)) {
		Region region = region_manager_.regionOf(region_id);
		if (!region.lease.valid()) continue;
		plan.request.mode = ExecutionMode::GpuOnly;
		plan.request.scope.regions = {region_id};
		plan.request.lease = region.lease;
		break;  // a single promoted Region is enough scope for now; multi-region
				// inserts are future work.
	}

	return plan;
}

Controller::RemovePlan Controller::routeRemove(VectorId id) {
	RemovePlan plan;
	plan.request.op = ModifyOp::Delete;
	plan.request.target = id;
	plan.request.mode = ExecutionMode::Hybrid;
	return plan;
}

TraverseResult Controller::dispatch(const TraverseRequest& request) {
	return scheduler_.schedule(request).get();
}

ModifyResult Controller::dispatch(const ModifyRequest& request) {
	return scheduler_.schedule(request).get();
}

SearchResult Controller::commitSearch(const SearchPlan& plan, const TraverseResult& result, bool final_was_hybrid) {
	// Registers this query vector as a new anchor candidate if not seen.
	routing_cache_.ensure(next_anchor_id_++, plan.primary.query.vector, kDefaultAnchorMaxDistance);
	recordTraversalForDrift(final_was_hybrid);
	SearchResult output = result.result;
	output.served_gpu_only = !final_was_hybrid;
	return output;
}

InsertResult Controller::commitInsert(const InsertPlan& plan, const ModifyResult& result) {
	if (result.ok && plan.anchor_id != 0) {
		promoteAnchor(plan.anchor_id, plan.request.record.vector, result.modified);
	}

	recordTraversalForDrift(plan.request.mode == ExecutionMode::Hybrid);
	return InsertResult{result.ok};
}

DeleteResult Controller::commitRemove(const RemovePlan& plan, const ModifyResult& result) {
	if (result.ok) {
		// Mirror image of commitInsert()'s promoteAnchor() call: a deleted
		// anchor's Region dependencies (if it ever had any -- evictAnchor() is
		// a no-op if not) no longer represent live data, so release them the
		// same way an eviction would (write back if dirty, free the device
		// allocation, stop tracking it for replacement) rather than leaving
		// them to look like a still-live anchor. Also drops it from
		// routing_cache_ so a future query/insert near this id's old vector
		// doesn't route to an anchor that no longer exists.
		evictAnchor(plan.request.target);
		routing_cache_.erase(plan.request.target);
	}
	recordTraversalForDrift(plan.request.mode == ExecutionMode::Hybrid);
	return DeleteResult{result.ok};
}

Controller::RoutingDecision Controller::route(const Query& query) {
	RoutingDecision decision;
	if (std::optional<VectorId> anchor_id = routing_cache_.nearest(query.vector)) {
		// Copied out of region_manager_ rather than referenced: it's guarded by
		// region_manager_'s own internal mutex, which can't outlive this call.
		std::vector<RegionId> regions = region_manager_.regionsOf(*anchor_id);
		if (!regions.empty()) {
			decision.gpu_only = true;
			decision.predicted_scope.regions = std::move(regions);
		}
	}
	return decision;
}

void Controller::recordTraversalForDrift(bool touched_host) {
	if (drift_window_total_ >= kDriftWindowSize) {
		drift_window_total_ = 0;
		drift_window_host_ = 0;
	}
	++drift_window_total_;
	if (touched_host) ++drift_window_host_;
}

void Controller::verify(const Query& query, VectorId anchor_id, const TraverseResult& gpu_only_result) {
	TraverseRequest verification_request{query, ExecutionMode::Hybrid, {}};
	TraverseResult verification_result = dispatch(verification_request);

	bool matched =
			gpu_only_result.result.neighbors.size() == verification_result.result.neighbors.size();
	for (std::size_t i = 0; matched && i < gpu_only_result.result.neighbors.size(); ++i) {
		matched = gpu_only_result.result.neighbors[i].id == verification_result.result.neighbors[i].id;
	}

	if (matched) return;

	// GPU-only diverged from ground truth: the regions this Anchor currently
	// depends on no longer represent its locality, so reclaim them all
	// (Quick Summary design point 3 feeding the point 4 replacement policy).
	ARACHNE_LOG_WARN("verification mismatch for anchor {}: reclaiming its region dependencies", anchor_id);
	evictAnchor(anchor_id);
}

std::optional<gpu::DeviceRegionHandle> Controller::allocateWithCompaction(std::size_t bytes) {
	std::optional<gpu::DeviceRegionHandle> handle = device_region_pool_.tryAllocate(bytes);
	if (handle.has_value()) return handle;

	if (!device_region_pool_.hasCapacity(bytes)) {
		return std::nullopt;  // genuinely over budget -- compacting frees nothing new
	}

	// hasCapacity() says `bytes` should fit under the budget, yet
	// tryAllocate() still failed: the live bytes are fragmented into holes
	// no single one of which is big enough. Consolidate once and retry
	// exactly once more (see allocateWithCompaction()'s doc comment,
	// controller.hpp).
	device_region_pool_.compact(gpu::MemoryKind::Data);
	stat_compactions_total_.fetch_add(1, std::memory_order_relaxed);
	return device_region_pool_.tryAllocate(bytes);
}

Controller::MakeResult Controller::make(VectorId anchor_id, RegionId region,
																				 std::vector<gpu::DeviceRegionPool::Lease>& pending,
																				 std::vector<std::vector<std::byte>>& zero_headers) {
	for (RegionId existing : region_manager_.regionsOf(anchor_id)) {
		if (existing == region) return MakeResult::Promoted;  // already a dependent
	}

	if (!region_manager_.isRegistered(region)) {
		ARACHNE_LOG_DEBUG("make: region {} is not registered, anchor {} cannot depend on it", region,
											 anchor_id);
		return MakeResult::NotEligible;
	}

	Region snapshot = region_manager_.regionOf(region);
	if (!snapshot.lease.valid()) {
		// Nobody has promoted this Region yet: acquire the one lease every
		// future dependent will share, and enqueue (but don't yet wait for)
		// its host-to-device copy -- promoteAnchor() flushes the whole batch
		// once every Region in the footprint has been attempted.
		IRegion* target = adapter_.resolveRegion(region);
		if (target == nullptr) return MakeResult::NotEligible;

		LeaseHandle lease = target->acquireWriteLease();
		if (!lease.valid()) {
			ARACHNE_LOG_DEBUG("make: region {} not lease-eligible for anchor {}", region, anchor_id);
			return MakeResult::NotEligible;
		}

		std::size_t header_bytes = gpu::DirtyHeaderBytes(snapshot.host.bytes, snapshot.host.subregion_bytes);
		std::optional<gpu::DeviceRegionHandle> device = allocateWithCompaction(header_bytes + snapshot.host.bytes);
		if (!device.has_value()) {
			// Can't hold a write lease over device memory that doesn't exist --
			// give it back so this isn't left half-promoted, and let the caller
			// (promoteAnchor()) decide whether evicting something and retrying
			// is worthwhile.
			target->releaseWriteLease(lease);
			ARACHNE_LOG_DEBUG("make: region {} ({} bytes) out of GPU capacity for anchor {}", region,
												 header_bytes + snapshot.host.bytes, anchor_id);
			return MakeResult::OutOfCapacity;
		}
		if (header_bytes > 0) {
			// Freshly allocated device memory is not guaranteed to be zeroed
			// (a pool arena can hand back bytes a previous, unrelated
			// allocation left behind) -- without this, a brand-new Region
			// could read back as already dirty purely by chance, defeating
			// writeBackDirtyRegions()'s whole point. Kept alive in
			// zero_headers (owned by promoteAnchor()'s call frame) until its
			// flush() actually lands the copy.
			zero_headers.emplace_back(header_bytes, std::byte{0});
			device_region_pool_.enqueueCopyFromHost(*device, zero_headers.back().data(), header_bytes,
																							 /*dst_offset=*/0, pending);
		}
		device_region_pool_.enqueueCopyFromHost(*device, snapshot.host.ptr, snapshot.host.bytes, header_bytes,
																						 pending);
		region_manager_.setDevice(region, *device);
		region_manager_.setLease(region, lease);
		stat_regions_promoted_.fetch_add(1, std::memory_order_relaxed);
	}
	// else: already promoted by some other Anchor -- anchor_id just becomes
	// an additional dependent of the lease (and GPU allocation) that's
	// already there, below.

	region_manager_.addDependency(anchor_id, region);
	replacement_policy_->onAnchorPromoted(anchor_id);

	ARACHNE_LOG_DEBUG("make: anchor {} now depends on region {}", anchor_id, region);
	return MakeResult::Promoted;
}

void Controller::promoteAnchor(VectorId anchor_id, const VectorView& anchor_vector,
																const RegionFootprint& footprint) {
	// Step 1: register this Anchor in the RoutingCache so future queries
	// close to it route to GPU (Quick Summary design point 1).
	routing_cache_.ensure(anchor_id, anchor_vector, kDefaultAnchorMaxDistance);

	// Every host-to-device copy make() enqueues below across the whole
	// footprint lands in one flush() at the end -- one scatter-gather round
	// trip for this Anchor's promotion, not one acquire+copy+sync per Region.
	// zero_headers keeps every synthesized zero-fill buffer alive until that
	// flush() actually runs.
	std::vector<gpu::DeviceRegionPool::Lease> pending;
	std::vector<std::vector<std::byte>> zero_headers;

	// Steps 2 + 3: grant a dependency on every region in scope, evicting via
	// replacement_policy_ (excluding anchor_id itself) if a region is out of
	// GPU capacity. Keeps trying the next victim -- not just one -- since a
	// single evicted Anchor's Regions may be smaller than what's needed here;
	// stops only once make() succeeds, reports a non-capacity reason (no
	// amount of eviction fixes that), or there's nobody left to reclaim.
	for (RegionId region : footprint.regions) {
		MakeResult result = make(anchor_id, region, pending, zero_headers);
		if (result == MakeResult::Promoted) continue;
		if (result == MakeResult::NotEligible) {
			ARACHNE_LOG_DEBUG("promoteAnchor: region {} not eligible for anchor {}, skipping", region,
												 anchor_id);
			continue;
		}

		while (result == MakeResult::OutOfCapacity) {
			std::optional<VectorId> victim = replacement_policy_->selectEvictionCandidate(anchor_id);
			if (!victim.has_value()) {
				ARACHNE_LOG_DEBUG(
						"promoteAnchor: no eviction candidate left, anchor {} region {} not promoted", anchor_id,
						region);
				break;
			}

			evictAnchor(*victim);
			result = make(anchor_id, region, pending, zero_headers);
		}

		if (result != MakeResult::Promoted) {
			ARACHNE_LOG_DEBUG("promoteAnchor: region {} still unavailable for anchor {} (result={})", region,
												 anchor_id, static_cast<int>(result));
		}
	}

	device_region_pool_.flush();
	// `pending`'s Leases (and zero_headers' scratch buffers) release here --
	// safe, flush() already proved every enqueued copy landed.
}

void Controller::evictAnchor(VectorId anchor_id) {
	// forget() only returns Regions that just dropped to zero dependents --
	// Regions some other Anchor still depends on are left promoted, since
	// they're still in use for a reason (see RegionManager::forget()'s doc
	// comment).
	std::vector<RegionId> orphaned = region_manager_.forget(anchor_id);

	// replacement_policy_ must stop tracking anchor_id here regardless of
	// whether forget() actually freed a Region: onAnchorEvicted() is
	// documented safe/idempotent to call even if anchor_id was never
	// tracked (see ReplacementPolicy's doc comment) or if its dependency
	// removal didn't happen to orphan anything (some other Anchor still
	// depends on the same Region). Calling this only inside the
	// !orphaned.empty() branch below was a real bug: an anchor evicted from
	// a multi-dependent Region would stay stuck at the front of FIFO's
	// tracked order forever (its RegionManager-side dependency is already
	// gone, so every future forget() on it is an immediate no-op that never
	// orphans anything either) -- promoteAnchor()'s eviction-retry loop
	// would then keep re-selecting that same already-forgotten anchor on
	// every iteration and never actually make progress freeing capacity.
	replacement_policy_->onAnchorEvicted(anchor_id);

	if (orphaned.empty()) return;

	std::vector<Region> snapshots;
	snapshots.reserve(orphaned.size());
	for (RegionId region_id : orphaned) {
		Region region = region_manager_.regionOf(region_id);  // still registered; residency not yet cleared
		if (IRegion* target = adapter_.resolveRegion(region_id)) {
			target->releaseWriteLease(region.lease);
		}
		snapshots.push_back(region);
	}

	std::vector<Region> resident;
	for (const Region& region : snapshots) {
		if (region.device.valid()) resident.push_back(region);
	}
	if (!resident.empty()) {
		writeBackDirtyRegions(resident);  // one batched gather across every Region here, not one per Region
		for (const Region& region : resident) device_region_pool_.free(region.device);
		stat_regions_evicted_.fetch_add(resident.size(), std::memory_order_relaxed);
	}

	for (RegionId region_id : orphaned) region_manager_.clearResidency(region_id);
	stat_anchor_evictions_.fetch_add(1, std::memory_order_relaxed);

	ARACHNE_LOG_DEBUG("evictAnchor: released anchor {}'s dependency on {} region(s)", anchor_id,
										 orphaned.size());
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

void Controller::writeBackDirtyRegions(const std::vector<Region>& regions) {
	const std::size_t n = regions.size();
	std::vector<std::size_t> header_bytes(n);
	std::vector<std::vector<std::byte>> header_buffers(n);
	std::vector<bool> dirty(n, true);  // conservative default for Regions with no header -- see doc comment

	// Phase 1: gather every Region's dirty-bitmap header in one batched,
	// async round trip -- one enqueue per Region that has a header, one
	// flush() for all of them together.
	{
		std::vector<gpu::DeviceRegionPool::Lease> pending;
		for (std::size_t i = 0; i < n; ++i) {
			header_bytes[i] = gpu::DirtyHeaderBytes(regions[i].host.bytes, regions[i].host.subregion_bytes);
			if (header_bytes[i] == 0) continue;  // no header to gather for this Region
			header_buffers[i].resize(header_bytes[i]);
			device_region_pool_.enqueueCopyToHost(regions[i].device, header_buffers[i].data(), header_bytes[i],
																						 /*src_offset=*/0, pending);
		}
		device_region_pool_.flush();
		// `pending`'s Leases release here -- safe, flush() already proved
		// every enqueued header copy landed.
	}

	for (std::size_t i = 0; i < n; ++i) {
		if (header_bytes[i] > 0) dirty[i] = AnyDirtyWordSet(header_buffers[i]);
	}

	// Phase 2: gather the actual data for every Region that needs writing
	// back (confirmed dirty, or no header to check -- see doc comment),
	// again as one batched, async round trip rather than one per Region.
	std::size_t written_back = 0;
	{
		std::vector<gpu::DeviceRegionPool::Lease> pending;
		for (std::size_t i = 0; i < n; ++i) {
			if (!dirty[i] || regions[i].host.ptr == nullptr || regions[i].host.bytes == 0) {
				ARACHNE_LOG_DEBUG("writeBackDirtyRegions: region {} is clean, skipping write-back", regions[i].id);
				continue;
			}
			device_region_pool_.enqueueCopyToHost(regions[i].device, regions[i].host.ptr, regions[i].host.bytes,
																						 header_bytes[i], pending);
			++written_back;
		}
		device_region_pool_.flush();
	}

	stat_regions_written_back_.fetch_add(written_back, std::memory_order_relaxed);
	ARACHNE_LOG_DEBUG("writeBackDirtyRegions: wrote back {} of {} region(s)", written_back, n);
}

void Controller::registerRegion(RegionId id, HostRegionView host) { region_manager_.registerRegion(id, host); }

RegionAccess Controller::acquireRegion(RegionId region) {
	Region snapshot = region_manager_.regionOf(region);  // throws if unregistered

	RegionAccess result;
	result.region = region;
	result.host = snapshot.host;
	if (snapshot.device.valid()) {
		result.on_device = true;
		result.device_lease.emplace(device_region_pool_.acquire(snapshot.device));
	}
	return result;
}

ControllerStats Controller::stats() const {
	ControllerStats result;
	result.regions_promoted_total = stat_regions_promoted_.load(std::memory_order_relaxed);
	result.regions_evicted_total = stat_regions_evicted_.load(std::memory_order_relaxed);
	result.regions_written_back_total = stat_regions_written_back_.load(std::memory_order_relaxed);
	result.anchor_evictions_total = stat_anchor_evictions_.load(std::memory_order_relaxed);
	result.compactions_total = stat_compactions_total_.load(std::memory_order_relaxed);
	result.gpu_bytes_allocated = device_region_pool_.bytesAllocated();
	return result;
}

}  // namespace arachne
