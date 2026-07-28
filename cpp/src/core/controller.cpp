#include "core/controller.hpp"

#include <cstddef>
#include <utility>

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
}  // namespace

#if defined(ARACHNE_WITH_RAFT)
Controller::Controller(IndexAdapter& adapter, RoutingCache& routing_cache, SchedulingConfig scheduling_config,
												std::unique_ptr<ReplacementPolicy> replacement_policy)
	: adapter_(adapter),
		routing_cache_(routing_cache),
		scheduler_(scheduling_config),
		replacement_policy_(std::move(replacement_policy)),
		stitch_pool_(device_) {
	if (replacement_policy_ == nullptr) replacement_policy_ = std::make_unique<FifoReplacementPolicy>();
	scheduler_.start(adapter_);
}
#else
Controller::Controller(IndexAdapter& adapter, RoutingCache& routing_cache, SchedulingConfig scheduling_config,
												std::unique_ptr<ReplacementPolicy> replacement_policy)
	: adapter_(adapter),
		routing_cache_(routing_cache),
		scheduler_(scheduling_config),
		replacement_policy_(std::move(replacement_policy)) {
	if (replacement_policy_ == nullptr) replacement_policy_ = std::make_unique<FifoReplacementPolicy>();
	scheduler_.start(adapter_);
}
#endif

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
	InsertPlan plan = routeInsert(record);
	ModifyResult result = dispatch(plan.request);
	return commitInsert(plan, result);
}

DeleteResult Controller::remove(VectorId id) {
	RemovePlan plan = routeRemove(id);
	ModifyResult result = dispatch(plan.request);
	return commitRemove(plan, result);
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

Controller::InsertPlan Controller::routeInsert(const Record& record) {
	InsertPlan plan;
	plan.anchor_id = record.id;

	plan.request.op = ModifyOp::Insert;
	plan.request.record = record;
	plan.request.mode = ExecutionMode::Hybrid;

	for (const Stitch& stitch : anchor_manager_.stitchesOf(plan.anchor_id)) {
		if (!stitch.lease.valid()) continue;
		plan.request.mode = ExecutionMode::GpuOnly;
		plan.request.scope.regions = {stitch.region};
		plan.request.lease = stitch.lease;
		break;  // a single active Stitch is enough scope for now; multi-region
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
	recordTraversalForDrift(plan.request.mode == ExecutionMode::Hybrid);
	return DeleteResult{result.ok};
}

Controller::RoutingDecision Controller::route(const Query& query) {
	RoutingDecision decision;
	if (std::optional<VectorId> anchor_id = routing_cache_.nearest(query.vector)) {
		// Copied out of anchor_manager_ rather than referenced: it's guarded by
		// anchor_manager_'s own internal mutex, which can't outlive this call.
		std::vector<Stitch> stitches = anchor_manager_.stitchesOf(*anchor_id);
		if (!stitches.empty()) {
			decision.gpu_only = true;
			decision.predicted_scope.regions.reserve(stitches.size());
			for (const Stitch& stitch : stitches) {
				decision.predicted_scope.regions.push_back(stitch.region);
			}
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

void Controller::promote(const RegionFootprint& hot_host_footprint) {
	// Placeholder policy: promote the whole reported footprint. A real policy
	// selects the minimal subset that satisfies GPU-only coverage across the
	// active Anchor working set (Quick Summary design point 2).
	for (RegionId id : hot_host_footprint.regions) {
		if (IRegion* region = adapter_.resolveRegion(id)) {
			region->materializeOnDevice();
		}
	}
}

void Controller::evict(const RegionFootprint& candidates) {
	// Placeholder policy: general read-side residency eviction, distinct from
	// the Stitch-based Lease replacement policy in anchor_manager_.
	for (RegionId id : candidates.regions) {
		if (IRegion* region = adapter_.resolveRegion(id)) {
			region->evictFromDevice();
		}
	}
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

	// GPU-only diverged from ground truth: the regions currently stitched to
	// this Anchor no longer represent its locality, so reclaim them all
	// (Quick Summary design point 3 feeding the point 4 replacement policy).
	ARACHNE_LOG_WARN("verification mismatch for anchor {}: reclaiming its stitches", anchor_id);
	evictAnchor(anchor_id);
}

bool Controller::make(VectorId anchor_id, RegionId region) {
	for (const Stitch& stitch : anchor_manager_.stitchesOf(anchor_id)) {
		if (stitch.region == region) return true;  // already stitched
	}

	IRegion* target = adapter_.resolveRegion(region);
	if (target == nullptr) return false;

	if (target->residency() != ResidencyState::Resident) {
		target->materializeOnDevice();  // Promotion step 2: copy the region's data onto GPU.
		if (target->residency() != ResidencyState::Resident) {
			ARACHNE_LOG_DEBUG("make: region {} did not become resident for anchor {}", region, anchor_id);
			return false;
		}
	}

	LeaseHandle lease = target->acquireWriteLease();
	if (!lease.valid()) {
		ARACHNE_LOG_DEBUG("make: region {} not lease-eligible for anchor {}", region, anchor_id);
		return false;
	}

	anchor_manager_.addStitch(anchor_id, region, lease);
	replacement_policy_->onStitchAdded(anchor_id);

	ARACHNE_LOG_DEBUG("make: stitched anchor {} to region {}", anchor_id, region);
	return true;
}

void Controller::promoteAnchor(VectorId anchor_id, const VectorView& anchor_vector,
																const RegionFootprint& footprint) {
	// Step 1: register this Anchor in the RoutingCache so future queries
	// close to it route to GPU (Quick Summary design point 1).
	routing_cache_.ensure(anchor_id, anchor_vector, kDefaultAnchorMaxDistance);

	// Steps 2 + 3: grant a Stitch for every region in scope, evicting via
	// replacement_policy_ (excluding anchor_id itself) if a region can't be
	// made available on the first attempt.
	for (RegionId region : footprint.regions) {
		if (make(anchor_id, region)) continue;

		std::optional<VectorId> victim = replacement_policy_->selectEvictionCandidate(anchor_id);
		if (!victim.has_value()) {
			ARACHNE_LOG_DEBUG(
					"promoteAnchor: no eviction candidate available, anchor {} region {} not promoted",
					anchor_id, region);
			continue;
		}

		evictAnchor(*victim);
		if (!make(anchor_id, region)) {
			ARACHNE_LOG_DEBUG(
					"promoteAnchor: region {} still unavailable for anchor {} after evicting anchor {}",
					region, anchor_id, *victim);
		}
	}
}

void Controller::evictAnchor(VectorId anchor_id) {
	std::vector<Stitch> stitches = anchor_manager_.forget(anchor_id);
	if (stitches.empty()) return;

	for (const Stitch& stitch : stitches) {
		if (IRegion* target = adapter_.resolveRegion(stitch.region)) {
			target->releaseWriteLease(stitch.lease);
		}
#if defined(ARACHNE_WITH_RAFT)
		// No-op until make() actually allocates through stitch_pool_ -- see its
		// doc comment. Guarded on validity so this is safe either way.
		if (stitch.memory.valid()) {
			stitch_pool_.free(stitch.memory);
		}
#endif
	}
	replacement_policy_->onAnchorEvicted(anchor_id);

	ARACHNE_LOG_DEBUG("evictAnchor: reclaimed {} stitch(es) from anchor {}", stitches.size(), anchor_id);
}

}  // namespace arachne
