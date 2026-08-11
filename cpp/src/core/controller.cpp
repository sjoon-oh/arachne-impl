#include "core/controller.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "logging.hpp"
#include "telemetry/trace.hpp"

namespace arachne {

namespace {
// Placeholder top_k for the lookup traversal insert() runs first. Core never
// interprets this beyond passing it through -- placement quality (HNSW's
// efConstruction or similar) is an adapter/index tuning concern the returned
// TraverseResult::hint is free to reflect however it needs to.
constexpr std::uint32_t kInsertionLookupTopK = 1;

// Bound once per OpScheduler execution worker thread (see class doc comment,
// controller.hpp); null on every other thread. acquireRegion() reads this to
// pick a worker's dedicated stream vs. the management stream.
thread_local cudaStream_t g_worker_stream = nullptr;
}  // namespace

Controller::Controller(IAdapter& adapter, RoutingCache& routing_cache, SchedulingConfig scheduling_config,
												std::unique_ptr<ReplacementPolicy> replacement_policy,
												std::size_t gpu_data_budget_bytes, std::size_t gpu_metadata_budget_bytes,
												std::size_t gpu_unit_bytes, std::unique_ptr<gpu::CompactionPolicy> compaction_policy,
												CoordinatorConfig coordinator_config, gpu::AllocationPolicy allocation_policy)
	: adapter_(adapter),
		routing_cache_(routing_cache),
		scheduler_(scheduling_config),
		device_(/*device_id=*/0, allocation_policy, gpu_data_budget_bytes, gpu_metadata_budget_bytes,
						scheduling_config.max_execution_threads, gpu_unit_bytes),
		device_region_pool_(device_, std::move(compaction_policy)),
		region_manager_(std::move(replacement_policy)) {
	scheduler_.start(
			adapter_, [this](std::size_t worker_index) { g_worker_stream = device_.workerStream(worker_index); },
			[this](TraverseRequest& request) {
				if (request.mode != ExecutionMode::GpuOnly) return;
				request.residency_pin = region_manager_.tryPinResidency(request.residency_hints);
				if (!request.residency_pin) {
					request.mode = ExecutionMode::Hybrid;
					request.scope = {};
				}
			},
			[this](ModifyRequest& request) {
				if (request.mode != ExecutionMode::GpuOnly) return;
				request.residency_pin = region_manager_.tryPinResidency(request.residency_hints);
				if (!request.residency_pin) {
					request.mode = ExecutionMode::Hybrid;
					request.scope = {};
					request.lease = LeaseHandle{};
				}
			});
	region_manager_.start(adapter_, device_region_pool_, routing_cache_, coordinator_config);
	ARACHNE_LOG_INFO(
			"Controller: started (gpu_data_budget={} gpu_metadata_budget={} gpu_unit_bytes={} "
			"max_execution_threads={})",
			gpu_data_budget_bytes, gpu_metadata_budget_bytes, gpu_unit_bytes, scheduling_config.max_execution_threads);
}

SearchResult Controller::search(const Query& query) {
	ARACHNE_TRACE_SCOPE("Controller", "search");
	SearchPlan plan = routeSearch(query);
	// Only mint/register an Anchor when this query needs a Hybrid
	// (host-driven) traversal -- a GpuOnly hit means existing residency
	// already answers it, so there's nothing for the replacement policy to
	// usefully consider promoting.
	VectorId anchor_id = (plan.primary.mode == ExecutionMode::Hybrid) ? next_anchor_id_.fetch_add(1) : 0;
	TraverseResult result = dispatch(plan.primary, anchor_id);
	bool final_was_hybrid = (result.execution_mode == ExecutionMode::Hybrid);

	if (plan.fallback_to_hybrid && !result.completed_within_scope) {
		TraverseRequest fallback_request{query, ExecutionMode::Hybrid, {}};
		anchor_id = next_anchor_id_.fetch_add(1);
		result = dispatch(fallback_request, anchor_id);
		final_was_hybrid = true;
	}

	return commitSearch(result, final_was_hybrid);
}

InsertResult Controller::insert(const Record& record) {
	ARACHNE_TRACE_SCOPE("Controller", "insert");
	// Claim record.id before doing anything else -- see insert()'s doc
	// comment (controller.hpp). The insert-and-check-.second pattern makes
	// two concurrent insert() calls for the same id race safely: exactly one
	// observes true and proceeds, the other sees false and bails out.
	{
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		if (!live_ids_.insert(record.id).second) {
			ARACHNE_LOG_WARN("insert: id {} is already live, rejecting duplicate insert", record.id);
			return InsertResult{false};
		}
	}

	try {
		// Step 1 (Traversal): find where this new vector belongs -- candidate
		// neighbors, a cluster to join, or whatever else the index's algorithm
		// needs (TraverseResult::hint) -- using the same anchor-routing decision
		// search() uses, so a repeatedly-inserted-near Anchor gets GpuOnly
		// lookups the same way a repeatedly-queried one does.
		Query lookup_query{record.vector, kInsertionLookupTopK};
		RoutingDecision decision = route(lookup_query);
		TraverseRequest lookup{lookup_query, decision.gpu_only ? ExecutionMode::GpuOnly : ExecutionMode::Hybrid,
													 decision.predicted_scope};
		// Unlike search(), always registers record.id as a promotion candidate
		// (it's always a brand-new Anchor here, never a redundant
		// re-registration) regardless of whether the Modify call below even
		// runs, let alone succeeds.
		TraverseResult candidates = dispatch(lookup, record.id);

		// Step 2 (Modification): apply the insert using what the traversal found.
		InsertPlan plan = routeInsert(record, std::move(candidates));
		ModifyResult result = dispatch(plan.request);
		InsertResult final_result = commitInsert(result);

		if (!final_result.ok) {
			// Never actually landed -- free the id back up rather than leaving
			// it permanently unusable.
			std::lock_guard<std::mutex> lock(live_ids_mutex_);
			live_ids_.erase(record.id);
		}
		return final_result;
	} catch (...) {
		// dispatch() can throw (e.g. a GpuOnly request reaching an adapter's
		// unimplemented traverseDevice()/modifyDevice(), see IAdapter's doc
		// comment) -- record.id was never actually inserted either way, so it
		// must not stay claimed.
		ARACHNE_LOG_WARN("insert: id {} threw during dispatch, releasing claimed id", record.id);
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		live_ids_.erase(record.id);
		throw;
	}
}

DeleteResult Controller::remove(VectorId id) {
	ARACHNE_TRACE_SCOPE("Controller", "remove");
	RemovePlan plan = routeRemove(id);
	ModifyResult result = dispatch(plan.request);
	DeleteResult final_result = commitRemove(plan, result);

	if (final_result.ok) {
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		live_ids_.erase(id);
	} else {
		ARACHNE_LOG_WARN("remove: id {} failed at the adapter, id remains live", id);
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
		plan.primary.residency_hints = decision.residency_hints;
		plan.fallback_to_hybrid = true;
	}
	return plan;
}

Controller::InsertPlan Controller::routeInsert(const Record& record, TraverseResult candidates) {
	InsertPlan plan;

	plan.request.op = ModifyOp::Insert;
	plan.request.record = record;
	plan.request.mode = ExecutionMode::Hybrid;
	// `candidates` isn't read again after this function, so its hint is moved
	// rather than copied -- see OpaqueData's doc comment for why Core carries
	// it without interpreting it.
	plan.request.scope = candidates.touched;
	plan.request.hint = std::move(candidates.hint);

	for (const RegionResidencyHint& hint : region_manager_.residencyHints(record.id)) {
		Region region = region_manager_.regionOf(hint.region);
		if (!region.lease.valid()) continue;
		plan.request.mode = ExecutionMode::GpuOnly;
		plan.request.scope.regions.clear();
		plan.request.scope.regions.push_back(hint.region);
		plan.request.lease = region.lease;
		plan.request.residency_hints.clear();
		plan.request.residency_hints.push_back(hint);
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

TraverseResult Controller::dispatch(const TraverseRequest& request, VectorId promotion_anchor_id) {
	ARACHNE_TRACE_SCOPE("Controller", "dispatchTraverse");
	// `vector` is captured now (copying just the VectorView struct, not its
	// backing bytes) rather than read from inside the lambda below --
	// OpScheduler's own copy of `request` only preserves the pointer, not its
	// lifetime, so this must happen while `request` (a reference into the
	// still-blocked caller's data) is still valid.
	VectorView vector = request.query.vector;
	std::future<TraverseResult> future =
			scheduler_.schedule(request, [this, promotion_anchor_id, vector](const TraverseResult& result) {
				region_manager_.recordTraversal(result.touched);
				if (promotion_anchor_id != 0) region_manager_.requestPromotion(promotion_anchor_id, result.touched, vector);
			});
	return future.get();
}

ModifyResult Controller::dispatch(const ModifyRequest& request) {
	ARACHNE_TRACE_SCOPE("Controller", "dispatchModify");
	return scheduler_.schedule(request).get();
}

SearchResult Controller::commitSearch(const TraverseResult& result, bool final_was_hybrid) {
	ARACHNE_TRACE_SCOPE("Controller", "commitSearch");
	SearchResult output = result.result;
	output.served_gpu_only = !final_was_hybrid;
	return output;
}

InsertResult Controller::commitInsert(const ModifyResult& result) {
	ARACHNE_TRACE_SCOPE("Controller", "commitInsert");
	return InsertResult{result.ok};
}

DeleteResult Controller::commitRemove(const RemovePlan& plan, const ModifyResult& result) {
	ARACHNE_TRACE_SCOPE("Controller", "commitRemove");
	if (result.ok) {
		// A deleted anchor's Region dependencies (if any -- releaseAnchor() is
		// a no-op otherwise) no longer represent live data; releaseAnchor()
		// also erases it from RoutingCache so a future query/insert near this
		// id's old vector doesn't route to an anchor that no longer exists.
		region_manager_.releaseAnchor(plan.request.target);
	}
	return DeleteResult{result.ok};
}

Controller::RoutingDecision Controller::route(const Query& query) {
	ARACHNE_TRACE_SCOPE("Controller", "route");
	RoutingDecision decision;
	if (std::optional<VectorId> anchor_id = routing_cache_.nearest(query.vector)) {
		// Copied out of region_manager_ rather than referenced: it's guarded by
		// region_manager_'s own internal mutex, which can't outlive this call.
		std::vector<RegionResidencyHint> hints = region_manager_.residencyHints(*anchor_id);
		if (!hints.empty()) {
			decision.gpu_only = true;
			decision.residency_hints = hints;
			decision.predicted_scope.regions.reserve(hints.size());
			for (const RegionResidencyHint& hint : hints) {
				decision.predicted_scope.regions.push_back(hint.region);
			}
		}
	}
	return decision;
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
	region_manager_.releaseAnchor(anchor_id);
}

void Controller::registerRegion(RegionId id, HostRegionView host) {
	ARACHNE_LOG_INFO("Controller::registerRegion: region {} ({} bytes host)", id, host.bytes);
	region_manager_.registerRegion(id, host);
}

RegionAccess Controller::acquireRegion(RegionId region) {
	Region snapshot = region_manager_.regionOf(region);  // throws if unregistered

	RegionAccess result;
	result.region = region;
	result.host = snapshot.host;
	result.residency_pin = region_manager_.tryPinResidency(
			{{region, snapshot.residency_generation}});
	if (result.residency_pin) {
		snapshot = region_manager_.regionOf(region);
		result.on_device = true;
		cudaStream_t stream = g_worker_stream != nullptr ? g_worker_stream : device_.managementStream();
		result.device_lease.emplace(device_region_pool_.acquire(snapshot.device, stream));
	}
	return result;
}

ControllerStats Controller::stats() const {
	RegionManager::Stats region_stats = region_manager_.stats();
	ControllerStats result;
	result.gpu_bytes_allocated = region_stats.gpu_bytes_allocated;
	result.regions_promoted_total = region_stats.regions_promoted_total;
	result.regions_evicted_total = region_stats.regions_evicted_total;
	result.regions_written_back_total = region_stats.regions_written_back_total;
	result.anchor_evictions_total = region_stats.anchor_evictions_total;
	result.compactions_total = region_stats.compactions_total;
	return result;
}

void Controller::waitIdle() {
	ARACHNE_LOG_INFO("Controller::waitIdle: forcing coordinator drain");
	region_manager_.waitIdle();
	ARACHNE_LOG_INFO("Controller::waitIdle: drain complete");
}

}  // namespace arachne
