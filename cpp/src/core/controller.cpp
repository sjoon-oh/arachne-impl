#include "core/controller.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "logging.hpp"

namespace arachne {

namespace {
// Placeholder: how many candidate neighbors the lookup traversal insert()
// runs first (see its doc comment) asks for. Core never interprets this
// beyond passing it through -- what actually matters for placement quality
// (HNSW's efConstruction or similar) is an adapter/index tuning concern the
// returned TraverseResult::hint is free to reflect however it needs to,
// regardless of this nominal top_k.
constexpr std::uint32_t kInsertionLookupTopK = 1;

// Bound to a real stream once, right at the top of each OpScheduler
// execution worker's thread (see the on_worker_start hook passed into
// scheduler_.start() below) -- null on every other thread (the thread
// calling insert()/search()/remove(), RegionManager's own Coordinator
// thread, a test's main thread, ...). Controller::acquireRegion() reads
// this to decide which of DeviceContext's streams a Lease it hands back
// should use: a worker's own dedicated compute stream if this thread is
// one, or the management stream otherwise -- see
// gpu::DeviceContext::workerStream()/managementStream()'s own doc comments
// for why the two are kept physically separate.
thread_local cudaStream_t g_worker_stream = nullptr;
}  // namespace

Controller::Controller(IAdapter& adapter, RoutingCache& routing_cache, SchedulingConfig scheduling_config,
												std::unique_ptr<ReplacementPolicy> replacement_policy,
												std::size_t gpu_data_budget_bytes, std::size_t gpu_metadata_budget_bytes,
												CoordinatorConfig coordinator_config)
	: adapter_(adapter),
		routing_cache_(routing_cache),
		scheduler_(scheduling_config),
		device_(/*device_id=*/0, gpu::AllocationPolicy::Pooled, gpu_data_budget_bytes, gpu_metadata_budget_bytes,
						scheduling_config.max_execution_threads),
		device_region_pool_(device_),
		region_manager_(std::move(replacement_policy)) {
	scheduler_.start(adapter_,
										[this](std::size_t worker_index) { g_worker_stream = device_.workerStream(worker_index); });
	region_manager_.start(adapter_, device_region_pool_, routing_cache_, coordinator_config);
}

SearchResult Controller::search(const Query& query) {
	SearchPlan plan = routeSearch(query);
	// An Anchor is only worth minting/registering when this query actually
	// needs a Hybrid (host-driven) traversal -- a GpuOnly hit means existing
	// residency already answers it, so there is nothing here for the
	// replacement policy to usefully consider promoting. See commitSearch()'s
	// own doc comment (controller.hpp).
	VectorId anchor_id = (plan.primary.mode == ExecutionMode::Hybrid) ? next_anchor_id_.fetch_add(1) : 0;
	TraverseResult result = dispatch(plan.primary, anchor_id);
	bool final_was_hybrid = (plan.primary.mode == ExecutionMode::Hybrid);

	if (plan.fallback_to_hybrid && !result.completed_within_scope) {
		TraverseRequest fallback_request{query, ExecutionMode::Hybrid, {}};
		anchor_id = next_anchor_id_.fetch_add(1);
		result = dispatch(fallback_request, anchor_id);
		final_was_hybrid = true;
	}

	return commitSearch(result, final_was_hybrid);
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
		// Unlike search(), always registers `record.id` as a promotion
		// candidate regardless of decision.gpu_only -- record.id is always a
		// brand-new Anchor (this is its first-ever insert), never a redundant
		// re-registration the way a repeatedly-queried search() hit would be.
		// Decoupled entirely from whether the Modify call below even runs, let
		// alone succeeds -- see commitInsert()'s own doc comment.
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

	for (RegionId region_id : region_manager_.regionsOf(record.id)) {
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

TraverseResult Controller::dispatch(const TraverseRequest& request, VectorId promotion_anchor_id) {
	// The one entry point search() and insert()'s own placement lookup both
	// go through. recordTraversal()/requestPromotion() run inside the
	// on_complete closure -- on the OpScheduler execution worker thread that
	// computed the result, not here on the calling thread after get()
	// unblocks -- see this method's own doc comment (controller.hpp) and
	// OpScheduler::schedule()'s on_complete doc comment for why. `vector` is
	// captured now (copying just the VectorView struct itself, not its
	// backing bytes) rather than reading `request.query.vector` from inside
	// the lambda -- request's own copy inside OpScheduler is only guaranteed
	// to preserve that pointer, not extend its lifetime, so this must be
	// read out on this thread while `request` (a reference into the still-
	// synchronously-blocked caller's own data) is still valid.
	VectorView vector = request.query.vector;
	std::future<TraverseResult> future =
			scheduler_.schedule(request, [this, promotion_anchor_id, vector](const TraverseResult& result) {
				region_manager_.recordTraversal(result.touched);
				if (promotion_anchor_id != 0) region_manager_.requestPromotion(promotion_anchor_id, result.touched, vector);
			});
	return future.get();
}

ModifyResult Controller::dispatch(const ModifyRequest& request) {
	return scheduler_.schedule(request).get();
}

SearchResult Controller::commitSearch(const TraverseResult& result, bool final_was_hybrid) {
	SearchResult output = result.result;
	output.served_gpu_only = !final_was_hybrid;
	return output;
}

InsertResult Controller::commitInsert(const ModifyResult& result) { return InsertResult{result.ok}; }

DeleteResult Controller::commitRemove(const RemovePlan& plan, const ModifyResult& result) {
	if (result.ok) {
		// Mirror image of insert()'s promotion request: a deleted anchor's
		// Region dependencies (if it ever had any -- releaseAnchor() is a
		// no-op if not) no longer represent live data, so release them (see
		// RegionManager::releaseAnchor()'s own doc comment for what's
		// immediate vs. lazy here) rather than leaving them to look like a
		// still-live anchor. releaseAnchor() also erases it from RoutingCache
		// itself now (see the class doc comment, region_manager.hpp) so a
		// future query/insert near this id's old vector doesn't route to an
		// anchor that no longer exists.
		region_manager_.releaseAnchor(plan.request.target);
	}
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

void Controller::registerRegion(RegionId id, HostRegionView host) { region_manager_.registerRegion(id, host); }

RegionAccess Controller::acquireRegion(RegionId region) {
	Region snapshot = region_manager_.regionOf(region);  // throws if unregistered

	RegionAccess result;
	result.region = region;
	result.host = snapshot.host;
	if (snapshot.device.valid()) {
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

void Controller::waitIdle() { region_manager_.waitIdle(); }

}  // namespace arachne
