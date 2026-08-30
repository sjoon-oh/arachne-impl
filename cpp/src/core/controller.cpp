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

// Reserved top bit of every id MintAnchorId() hands out -- see
// next_anchor_id_'s own doc comment (controller.hpp) for why this needs to
// be structurally impossible to collide with, not just unlikely to: an
// adapter's own VectorId space (base + inserted elements) would need to
// exceed 2^63 entries to ever reach this bit on its own, which is not a
// realistic concern for any deployment of this system.
constexpr VectorId kAnchorIdBit = VectorId{1} << 63;

// Bound once per OpScheduler execution worker thread (see class doc comment,
// controller.hpp); null on every other thread. acquireRegion() reads this to
// pick a worker's dedicated stream vs. the management stream.
thread_local cudaStream_t g_worker_stream = nullptr;

// Bound alongside g_worker_stream above, same lifetime/thread-affinity;
// null on every other thread, or if the adapter requested no scratch (see
// IAdapter::requiredScratchBytesPerWorker()). workerScratch() reads this.
thread_local void* g_worker_scratch = nullptr;
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
	// Before any worker thread starts (reserveWorkerScratch() itself throws
	// if called any later) -- see IAdapter::requiredScratchBytesPerWorker()'s
	// doc comment. 0 is a cheap no-op: most adapters need nothing here.
	device_.reserveWorkerScratch(adapter_.requiredScratchBytesPerWorker());
	scheduler_.start(
			adapter_,
			[this](std::size_t worker_index) {
				g_worker_stream = device_.workerStream(worker_index);
				g_worker_scratch = device_.workerScratch(worker_index);
			},
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
	// Resolve CoordinatorConfig's fraction-of-budget convenience fields (see
	// their own doc comment) against device_'s *actual* budget -- only
	// reachable here, after device_ has already finished construction above,
	// since that's the only place the real, possibly unit-rounded
	// (AllocationPolicy::Pooled) capacity is known. Left untouched (whatever
	// the caller set max_promotion_bytes_per_pass/max_eviction_bytes_per_pass
	// to directly) when the corresponding fraction is std::nullopt.
	const std::size_t data_budget_bytes = device_.budgetBytes(gpu::MemoryKind::Data);
	if (coordinator_config.max_promotion_fraction_of_budget) {
		coordinator_config.max_promotion_bytes_per_pass = static_cast<std::size_t>(
				static_cast<double>(data_budget_bytes) * *coordinator_config.max_promotion_fraction_of_budget);
	}
	if (coordinator_config.max_eviction_fraction_of_budget) {
		coordinator_config.max_eviction_bytes_per_pass = static_cast<std::size_t>(
				static_cast<double>(data_budget_bytes) * *coordinator_config.max_eviction_fraction_of_budget);
	}
	region_manager_.start(adapter_, device_region_pool_, routing_cache_, coordinator_config);
	ARACHNE_LOG_INFO(
			"Controller: started (gpu_data_budget={} gpu_metadata_budget={} gpu_unit_bytes={} "
			"max_execution_threads={} max_promotion_bytes_per_pass={} max_eviction_bytes_per_pass={})",
			gpu_data_budget_bytes, gpu_metadata_budget_bytes, gpu_unit_bytes, scheduling_config.max_execution_threads,
			coordinator_config.max_promotion_bytes_per_pass, coordinator_config.max_eviction_bytes_per_pass);
}

SearchResult Controller::search(const Query& query, bool record_for_replacement_policy) {
	ARACHNE_TRACE_SCOPE("Controller", "search");
	return submitSearch(query, record_for_replacement_policy).get();
}

InsertResult Controller::insert(const Record& record) {
	ARACHNE_TRACE_SCOPE("Controller", "insert");
	try {
		return submitInsert(record).get();
	} catch (...) {
		// submitInsert()'s own chained callbacks already released record.id
		// back out of live_ids_ before this exception could reach here (see
		// its doc comment) -- this catch exists purely to preserve the log
		// message dispatch()-throwing used to produce synchronously.
		ARACHNE_LOG_WARN("insert: id {} threw during dispatch, releasing claimed id", record.id);
		throw;
	}
}

DeleteResult Controller::remove(VectorId id) {
	ARACHNE_TRACE_SCOPE("Controller", "remove");
	return submitRemove(id).get();
}

VectorId Controller::MintAnchorId() {
	// CAS retry rather than a bare fetch_add(): the only way this counter's
	// raw value could ever repeat a previously-minted one is wraparound --
	// completing a full lap of every representable std::uint64_t value
	// (2^64 of them), not merely growing past 2^63 (OR-ing the top bit onto
	// a payload that already has it set, once the counter itself climbs
	// that high, is harmless -- see next_anchor_id_'s own doc comment). This
	// loop handles two distinct things that can happen on the way to a
	// wraparound, in order:
	//
	//  1. The wrap step itself lands on payload 0, which collides with 0's
	//     reserved "no Anchor" meaning (see TraverseRequest::anchor_id /
	//     RoutingDecision::anchor_id, both std::optional<VectorId>, and
	//     every `anchor_id != 0`/`!lookup.anchor_id` check throughout this
	//     file) -- rewritten to 1 unconditionally, a cheap, always-correct
	//     guard independent of how unrealistic reaching it actually is.
	//  2. Every payload from here on (this is the *second* lap now) is a
	//     candidate for colliding with some Anchor id minted during the
	//     first lap that's still meaningful to something -- still resident
	//     (RegionManager::isKnownAnchor()'s dependencies_ check) or released
	//     but permanently remembered anyway (its anchor_epoch_ check, see
	//     that member's own doc comment on why it's never erased). Checked
	//     -- and, on an actual collision, retried with the next payload
	//     instead of reissuing it -- only once next_anchor_id_wrapped_ is
	//     ever observed true, so every mint before the first (if ever) full
	//     lap stays entirely lock-free on this path; RegionManager's own
	//     mutex only gets pulled into a request this rare.
	VectorId current = next_anchor_id_.load(std::memory_order_relaxed);
	VectorId payload;
	for (;;) {
		bool wrapped_this_step;
		do {
			payload = current + 1;
			wrapped_this_step = (payload == 0);
			if (wrapped_this_step) payload = 1;
		} while (!next_anchor_id_.compare_exchange_weak(current, payload, std::memory_order_relaxed));
		if (wrapped_this_step) next_anchor_id_wrapped_.store(true, std::memory_order_relaxed);

		if (!next_anchor_id_wrapped_.load(std::memory_order_relaxed)) return kAnchorIdBit | payload;
		if (!region_manager_.isKnownAnchor(kAnchorIdBit | payload)) return kAnchorIdBit | payload;
		// Collision on a second (or later) lap -- try the very next payload
		// rather than this one. `current` already equals `payload` here
		// (this iteration's successful CAS just set next_anchor_id_ to it),
		// so the next do-while naturally continues counting forward from it.
		current = payload;
	}
}

std::future<SearchResult> Controller::submitSearch(const Query& query, bool record_for_replacement_policy) {
	ARACHNE_TRACE_SCOPE("Controller", "submitSearch");
	SearchPlan plan = routeSearch(query);
	// Only mint/register an Anchor when this query needs a Hybrid
	// (host-driven) traversal -- a GpuOnly hit means existing residency
	// already answers it, so there's nothing for the replacement policy to
	// usefully consider promoting.
	VectorId anchor_id = (plan.primary.mode == ExecutionMode::Hybrid) ? MintAnchorId() : 0;
	// If routeSearch() didn't already find an existing Anchor for this query
	// (plan.primary.anchor_id still empty), this freshly minted id is what
	// requestPromotion() below registers this locality under -- give the
	// adapter that same identity up front so a Hybrid traversal completing
	// this call can populate its own entry-point cache (see
	// TraverseRequest::anchor_id's doc comment) against the exact id a later
	// query landing on this same, now-registered Anchor will be routed with.
	// Left untouched when routeSearch() already found a real Anchor (a
	// different id would just discard it).
	if (anchor_id != 0 && !plan.primary.anchor_id) plan.primary.anchor_id = anchor_id;

	// A shared_ptr, not a local std::promise, because it must outlive this
	// function (which returns immediately) and be safely resolved from
	// whichever worker thread's on_complete ends up settling it -- possibly
	// the fallback retry's callback below, running later on a different
	// batch than the primary dispatch.
	auto result_promise = std::make_shared<std::promise<SearchResult>>();
	std::future<SearchResult> result_future = result_promise->get_future();

	bool fallback_to_hybrid = plan.fallback_to_hybrid;
	// `query` itself isn't captured (see submitSearch()'s doc comment on
	// caller-owned vector lifetime) -- only its VectorView/top_k, exactly
	// what a fallback retry request needs to rebuild.
	Query query_view = query;

	scheduler_.schedule(plan.primary,
			[this, result_promise, anchor_id, record_for_replacement_policy, fallback_to_hybrid, query_view](
					std::exception_ptr error, const TraverseResult& first_result) {
				if (error) {
					result_promise->set_exception(error);
					return;
				}
				if (record_for_replacement_policy) {
					region_manager_.recordTraversal(first_result.touched);
					if (anchor_id != 0) {
						region_manager_.requestPromotion(anchor_id, first_result.touched, query_view.vector);
					}
				}
				bool first_was_hybrid = (first_result.execution_mode == ExecutionMode::Hybrid);

				if (!fallback_to_hybrid || first_result.completed_within_scope) {
					result_promise->set_value(commitSearch(first_result, first_was_hybrid));
					return;
				}

				// GpuOnly attempt fell short of its predicted scope -- retry
				// Hybrid, exactly like the old synchronous search() did, just
				// chained via on_complete instead of blocking this worker thread
				// on a second future (see TraverseTask::on_complete's doc comment
				// on why: with a single execution worker, blocking here for the
				// retry's own batch to be picked up would deadlock).
				TraverseRequest fallback_request{query_view, ExecutionMode::Hybrid, {}};
				VectorId retry_anchor_id = MintAnchorId();
				fallback_request.anchor_id = retry_anchor_id;
				scheduler_.schedule(fallback_request,
						[this, result_promise, retry_anchor_id, record_for_replacement_policy, query_view](
								std::exception_ptr retry_error, const TraverseResult& retry_result) {
							if (retry_error) {
								result_promise->set_exception(retry_error);
								return;
							}
							if (record_for_replacement_policy) {
								region_manager_.recordTraversal(retry_result.touched);
								region_manager_.requestPromotion(retry_anchor_id, retry_result.touched, query_view.vector);
							}
							result_promise->set_value(commitSearch(retry_result, /*final_was_hybrid=*/true));
						});
			});

	return result_future;
}

std::future<InsertResult> Controller::submitInsert(const Record& record) {
	ARACHNE_TRACE_SCOPE("Controller", "submitInsert");
	// Claim record.id before doing anything else -- see insert()'s doc
	// comment (controller.hpp). The insert-and-check-.second pattern makes
	// two concurrent insert() calls for the same id race safely: exactly one
	// observes true and proceeds, the other sees false and bails out.
	{
		std::lock_guard<std::mutex> lock(live_ids_mutex_);
		if (!live_ids_.insert(record.id).second) {
			ARACHNE_LOG_WARN("submitInsert: id {} is already live, rejecting duplicate insert", record.id);
			std::promise<InsertResult> rejected;
			rejected.set_value(InsertResult{false});
			return rejected.get_future();
		}
	}

	auto result_promise = std::make_shared<std::promise<InsertResult>>();
	std::future<InsertResult> result_future = result_promise->get_future();

	// Step 1 (Traversal): find where this new vector belongs -- candidate
	// neighbors, a cluster to join, or whatever else the index's algorithm
	// needs (TraverseResult::hint) -- using the same anchor-routing decision
	// search() uses, so a repeatedly-inserted-near Anchor gets GpuOnly
	// lookups the same way a repeatedly-queried one does.
	Query lookup_query{record.vector, kInsertionLookupTopK};
	RoutingDecision decision = route(lookup_query);
	TraverseRequest lookup{lookup_query, decision.gpu_only ? ExecutionMode::GpuOnly : ExecutionMode::Hybrid,
												 decision.predicted_scope};
	lookup.anchor_id = decision.anchor_id;
	// Mint a fresh Anchor id when routeInsert()/route() didn't already find
	// an existing one nearby -- exactly like search()'s own fallback below,
	// not record.id: an Anchor's id and lifecycle are Core's own bookkeeping,
	// independent of whether *this specific* insert's vector is what
	// originally caused it to exist (see MintAnchorId()'s own doc comment,
	// and next_anchor_id_'s in controller.hpp, for why reusing record.id
	// here used to conflate the two and what that broke: deleting this
	// vector later must not be assumed to mean "this Anchor is done").
	if (!lookup.anchor_id) lookup.anchor_id = MintAnchorId();
	// Whatever this lookup ended up tagged with (an existing nearby Anchor,
	// or the freshly-minted one above) -- *not* record_id below, now that
	// the two are no longer the same thing. requestPromotion() must register
	// this traversal's footprint under the same Anchor id the traversal
	// itself ran under, or RegionManager/the ReplacementPolicy would track
	// dependencies against an id nothing else ever refers to.
	VectorId insert_anchor_id = *lookup.anchor_id;

	// `record` itself isn't captured (non-owning VectorView, see
	// submitSearch()'s doc comment on the same issue) -- record.id is a
	// plain value, safe regardless.
	VectorId record_id = record.id;
	Record record_by_value = record;

	// Always registers a promotion candidate for insert_anchor_id (regardless
	// of whether the Modify call below even runs, let alone succeeds) --
	// unlike search(), there's no "GpuOnly hit, nothing to promote" case
	// here, insert()'s lookup traversal always runs Hybrid or GpuOnly-with-
	// residency-hints, either way worth recording.
	scheduler_.schedule(lookup,
			[this, result_promise, insert_anchor_id, record_id, record_by_value](std::exception_ptr error,
					const TraverseResult& candidates) {
				if (error) {
					// dispatch() throwing here used to leave record.id un-claimed
					// (insert()'s catch block released it) -- do the same before
					// propagating, so a failed lookup never permanently blocks
					// this id from ever being inserted again.
					std::lock_guard<std::mutex> lock(live_ids_mutex_);
					live_ids_.erase(record_id);
					result_promise->set_exception(error);
					return;
				}
				region_manager_.recordTraversal(candidates.touched);
				region_manager_.requestPromotion(insert_anchor_id, candidates.touched, record_by_value.vector);

				// Step 2 (Modification): apply the insert using what the
				// traversal found. `candidates` is copied (routeInsert() takes it
				// by value and moves its `hint` out) since this callback only has
				// a const&.
				InsertPlan plan = routeInsert(record_by_value, candidates);
				scheduler_.schedule(plan.request,
						[this, result_promise, record_id](std::exception_ptr modify_error, const ModifyResult& result) {
							if (modify_error) {
								std::lock_guard<std::mutex> lock(live_ids_mutex_);
								live_ids_.erase(record_id);
								result_promise->set_exception(modify_error);
								return;
							}
							InsertResult final_result = commitInsert(result);
							if (!final_result.ok) {
								// Never actually landed -- free the id back up rather than
								// leaving it permanently unusable.
								std::lock_guard<std::mutex> lock(live_ids_mutex_);
								live_ids_.erase(record_id);
							}
							result_promise->set_value(final_result);
						});
			});

	return result_future;
}

std::future<DeleteResult> Controller::submitRemove(VectorId id) {
	ARACHNE_TRACE_SCOPE("Controller", "submitRemove");
	RemovePlan plan = routeRemove(id);

	auto result_promise = std::make_shared<std::promise<DeleteResult>>();
	std::future<DeleteResult> result_future = result_promise->get_future();

	scheduler_.schedule(plan.request,
			[this, result_promise, plan, id](std::exception_ptr error, const ModifyResult& result) {
				if (error) {
					result_promise->set_exception(error);
					return;
				}
				DeleteResult final_result = commitRemove(plan, result);
				if (final_result.ok) {
					std::lock_guard<std::mutex> lock(live_ids_mutex_);
					live_ids_.erase(id);
				} else {
					ARACHNE_LOG_WARN("remove: id {} failed at the adapter, id remains live", id);
				}
				result_promise->set_value(final_result);
			});

	return result_future;
}

Controller::SearchPlan Controller::routeSearch(const Query& query) {
	RoutingDecision decision = route(query);

	SearchPlan plan;
	plan.primary.query = query;
	plan.primary.mode = ExecutionMode::Hybrid;
	plan.primary.scope = {};
	plan.primary.anchor_id = decision.anchor_id;
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

TraverseResult Controller::dispatch(const TraverseRequest& request, VectorId promotion_anchor_id,
		bool record_traversal) {
	ARACHNE_TRACE_SCOPE("Controller", "dispatchTraverse");
	return dispatchAsync(request, promotion_anchor_id, record_traversal).get();
}

std::future<TraverseResult> Controller::dispatchAsync(
		const TraverseRequest& request, VectorId promotion_anchor_id, bool record_traversal) {
	ARACHNE_TRACE_SCOPE("Controller", "dispatchTraverseAsync");
	// `vector` is captured now (copying just the VectorView struct, not its
	// backing bytes) rather than read from inside the lambda below --
	// OpScheduler's own copy of `request` only preserves the pointer, not its
	// lifetime, so this must happen while `request` (a reference into the
	// still-live caller's data) is still valid -- true for dispatch()'s
	// blocking callers, and for dispatchAsync()'s own callers as long as
	// they honor the same lifetime contract submitSearch()/submitInsert()'s
	// doc comments describe.
	VectorView vector = request.query.vector;
	return scheduler_.schedule(
			request, [this, promotion_anchor_id, record_traversal, vector](std::exception_ptr error, const TraverseResult& result) {
				if (error || !record_traversal) return;
				region_manager_.recordTraversal(result.touched);
				if (promotion_anchor_id != 0) region_manager_.requestPromotion(promotion_anchor_id, result.touched, vector);
			});
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

DeleteResult Controller::commitRemove(const RemovePlan&, const ModifyResult& result) {
	ARACHNE_TRACE_SCOPE("Controller", "commitRemove");
	// Deliberately does *not* call region_manager_.releaseAnchor() here
	// anymore -- plan.request.target is the deleted *data* element's own id,
	// which is no longer assumed to double as some Anchor's id too (see
	// MintAnchorId()'s and next_anchor_id_'s own doc comments for why that
	// assumption was wrong in general, and the latency-tracing report entry,
	// cpp/test/index/report/, for the measured cost of every delete taking
	// CostAwareReplacementPolicy::mutex_ to chase a relationship that mostly
	// didn't exist). An Anchor's own lifecycle -- whether it's still worth
	// keeping GPU-resident -- is entirely the ReplacementPolicy's call now,
	// driven by heat decay and capacity pressure like any other Anchor,
	// independent of whatever data element happened to trigger its creation.
	// Controller::verify() still calls releaseAnchor() directly, with a real
	// Anchor id, for the one case that's still Core's call to make: a
	// verification mismatch means that Anchor's current Region dependencies
	// no longer represent its locality at all, regardless of heat.
	return DeleteResult{result.ok};
}

Controller::RoutingDecision Controller::route(const Query& query) {
	ARACHNE_TRACE_SCOPE("Controller", "route");
	RoutingDecision decision;
	if (std::optional<VectorId> anchor_id = routing_cache_.nearest(query.vector)) {
		decision.anchor_id = anchor_id;
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

void* Controller::workerScratch() const { return g_worker_scratch; }

cudaStream_t Controller::workerStream() const {
	return g_worker_stream != nullptr ? g_worker_stream : device_.managementStream();
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
	result.relocation_batches_total = region_stats.relocation_batches_total;
	result.candidates_requeued_total = region_stats.candidates_requeued_total;
	result.candidates_rejected_total = region_stats.candidates_rejected_total;
	return result;
}

void Controller::waitIdle() {
	ARACHNE_LOG_INFO("Controller::waitIdle: forcing coordinator drain");
	region_manager_.waitIdle();
	ARACHNE_LOG_INFO("Controller::waitIdle: drain complete");
}

}  // namespace arachne
