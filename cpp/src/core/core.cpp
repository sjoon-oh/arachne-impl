#include "arachne/core/core.hpp"

#include <cstddef>

#include "arachne/logging.hpp"

namespace arachne {

namespace {
constexpr std::uint64_t kDriftWindowSize = 128;
}  // namespace

Core::Core(IndexAdapter& adapter, RoutingCache& routing_cache)
	: adapter_(adapter), routing_cache_(routing_cache) {}

SearchResult Core::search(const Query& query) {
	RoutingDecision decision = route(query);

	TraversalRequest request{query, ExecutionMode::Hybrid, {}};
	if (decision.gpu_only_eligible) {
		request.mode = ExecutionMode::GpuOnly;
		request.scope = decision.predicted_scope;
	}

	TraversalResult result = adapter_.traverse(request);

	if (decision.gpu_only_eligible && !result.completed_within_scope) {
		// GPU-only path couldn't stay within scope; fall back to hybrid so the
		// caller still gets a complete answer (Quick Summary design point 1).
		request.mode = ExecutionMode::Hybrid;
		result = adapter_.traverse(request);
	}

	recordTraversalForDrift(request.mode == ExecutionMode::Hybrid);
	// Registers this vector's Anchor for future matching if it isn't already
	// known.
	routing_cache_.ensure(next_anchor_id_++, query.vector);

	return result.result;
}

InsertResult Core::insert(const Record& record) {
	VectorId anchor_id = routing_cache_.ensure(next_anchor_id_++, record.vector);

	ModificationRequest request;
	request.op = ModificationOp::Insert;
	request.record = record;
	request.mode = ExecutionMode::Hybrid;

	for (const Stitch& stitch : anchor_manager_.stitchesOf(anchor_id)) {
		if (!stitch.lease.valid()) continue;
		request.mode = ExecutionMode::GpuOnly;
		request.scope.regions = {stitch.region};
		request.lease = stitch.lease;
		break;  // a single active Stitch is enough scope for now; multi-region
						// inserts are future work.
	}

	ModificationResult result = adapter_.modify(request);

	if (result.ok) {
		for (RegionId region : result.modified.regions) {
			make(anchor_id, region);
		}
	}

	recordTraversalForDrift(request.mode == ExecutionMode::Hybrid);
	return InsertResult{result.ok};
}

DeleteResult Core::remove(VectorId id) {
	ModificationRequest request;
	request.op = ModificationOp::Delete;
	request.target = id;
	request.mode = ExecutionMode::Hybrid;

	ModificationResult result = adapter_.modify(request);
	return DeleteResult{result.ok};
}

Core::RoutingDecision Core::route(const Query& query) const {
	RoutingDecision decision;
	if (std::optional<VectorId> anchor_id = routing_cache_.nearest(query.vector)) {
		std::vector<Stitch> stitches = anchor_manager_.stitchesOf(*anchor_id);
		if (!stitches.empty()) {
			decision.gpu_only_eligible = true;
			for (const Stitch& stitch : stitches) {
				decision.predicted_scope.regions.push_back(stitch.region);
			}
		}
	}
	return decision;
}

void Core::recordTraversalForDrift(bool touched_host) {
	if (drift_window_total_ >= kDriftWindowSize) {
		drift_window_total_ = 0;
		drift_window_host_ = 0;
	}
	++drift_window_total_;
	if (touched_host) ++drift_window_host_;
}

void Core::promote(const RegionFootprint& hot_host_footprint) {
	// Placeholder policy: promote the whole reported footprint. A real policy
	// selects the minimal subset that satisfies GPU-only coverage across the
	// active Anchor working set (Quick Summary design point 2).
	for (RegionId id : hot_host_footprint.regions) {
		if (IRegion* region = adapter_.resolveRegion(id)) {
			region->materializeOnDevice();
		}
	}
}

void Core::evict(const RegionFootprint& candidates) {
	// Placeholder policy: general read-side residency eviction, distinct from
	// the Stitch-based Lease replacement policy in anchor_manager_.
	for (RegionId id : candidates.regions) {
		if (IRegion* region = adapter_.resolveRegion(id)) {
			region->evictFromDevice();
		}
	}
}

void Core::verify(const Query& query, VectorId anchor_id, const TraversalResult& gpu_only_result) {
	TraversalRequest verification_request{query, ExecutionMode::Hybrid, {}};
	TraversalResult verification_result = adapter_.traverse(verification_request);

	bool matched =
			gpu_only_result.result.neighbors.size() == verification_result.result.neighbors.size();
	for (std::size_t i = 0; matched && i < gpu_only_result.result.neighbors.size(); ++i) {
		matched = gpu_only_result.result.neighbors[i].id == verification_result.result.neighbors[i].id;
	}

	if (matched) return;

	// GPU-only diverged from ground truth: the regions currently stitched to
	// this Anchor no longer represent its locality, so reclaim them all
	// (Quick Summary design point 3 feeding the point 4 replacement policy).
	std::vector<Stitch> stale = anchor_manager_.forget(anchor_id);

	ARACHNE_LOG_WARN("verification mismatch for anchor {}: reclaiming {} stitch(es)", anchor_id,
										stale.size());
	for (const Stitch& stitch : stale) {
		if (IRegion* target = adapter_.resolveRegion(stitch.region)) {
			target->releaseWriteLease(stitch.lease);
		}
	}
}

bool Core::make(VectorId anchor_id, RegionId region) {
	for (const Stitch& stitch : anchor_manager_.stitchesOf(anchor_id)) {
		if (stitch.region == region) return true;  // already stitched
	}

	IRegion* target = adapter_.resolveRegion(region);
	if (target == nullptr || target->residency() != ResidencyState::Resident) return false;

	LeaseHandle lease = target->acquireWriteLease();
	if (!lease.valid()) {
		ARACHNE_LOG_DEBUG("make: region {} not lease-eligible for anchor {}", region, anchor_id);
		return false;
	}

	anchor_manager_.addStitch(anchor_id, region, lease);

	ARACHNE_LOG_DEBUG("make: stitched anchor {} to region {}", anchor_id, region);
	return true;
}

}  // namespace arachne
