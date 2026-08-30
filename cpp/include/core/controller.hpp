#pragma once

#include <atomic>
#include <optional>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "adapter/index_adapter.hpp"
#include "core/op_scheduler.hpp"
#include "core/region_manager.hpp"
#include "core/replacement_policy.hpp"
#include "core/routing_cache.hpp"
#include "gpu/device_context.hpp"
#include "gpu/device_region_pool.hpp"

namespace arachne {

/// Result of Controller::acquireRegion(): tells the caller where `region`'s
/// data currently lives so it can decide how to touch it -- kernel-launch
/// against `device_lease` if `on_device` is true, or handle it itself via
/// `host` (always populated; Arachne never owns host allocation, see
/// HostRegionView's doc comment) if not. This is the replacement for the
/// old, removed gpu::DeviceRegionPool::access(): a single acquire-shaped
/// entry point that answers "where is this" and, when the answer is
/// "device", hands back a Lease guaranteeing the pointer stays valid for as
/// long as the caller holds onto this result -- rather than a bare pointer
/// with no such guarantee.
///
/// Enforcement caveat (same one that applies to gpu::DeviceRegionPool::Lease
/// itself): this is a contract, not something the compiler enforces. An
/// adapter that reads `host.ptr` directly without going through this call
/// isn't stopped from doing so -- acquireRegion() exists to give a
/// cooperating caller a single place to ask "GPU or host, and is it safe
/// right now", not to make bypassing it impossible.
struct RegionAccess {
	RegionId region = 0;
	HostRegionView host;
	bool on_device = false;
	std::shared_ptr<void> residency_pin;
	std::optional<gpu::DeviceRegionPool::Lease> device_lease;  // engaged iff on_device
};

/// Snapshot of Controller::stats(): structured, queryable state -- a thin
/// copy of RegionManager::Stats (see its own doc comment), which is where
/// GPU residency management -- and its counters -- actually live now.
/// Kept as Controller's own type for API stability. `gpu_bytes_allocated`
/// is a live read (delegates to gpu::DeviceRegionPool::bytesAllocated());
/// the four `_total` fields are monotonically increasing counters for the
/// Controller's whole lifetime, not point-in-time state -- e.g.
/// `regions_evicted_total` counts every Region ever reclaimed, not how many
/// are currently evicted (there's no such thing as "currently evicted": an
/// evicted Region is just back to host-only, indistinguishable from one
/// never promoted).
struct ControllerStats {
	std::size_t gpu_bytes_allocated = 0;
	std::uint64_t regions_promoted_total = 0;       // RegionManager::make() calls that newly acquired a lease
	std::uint64_t regions_evicted_total = 0;        // Regions reclaimed via RegionManager (any anchor)
	std::uint64_t regions_written_back_total = 0;   // of those, how many actually had data copied back
	std::uint64_t anchor_evictions_total = 0;       // anchor releases that reclaimed >=1 Region
	std::uint64_t compactions_total = 0;            // RegionManager falling back to gpu::DeviceRegionPool::compact()
	std::uint64_t relocation_batches_total = 0;     // buildRelocationPlan()/processRelocationBatch() passes run
	std::uint64_t candidates_requeued_total = 0;    // promotion candidates pushed back to the policy for a later pass
																									 // (crowded out by a same-pass cap/budget, or a transient
																									 // execution-time failure -- see region_manager.cpp's
																									 // buildRelocationPlan()/processRelocationBatch())
	std::uint64_t candidates_rejected_total = 0;    // promotion candidates a ReplacementPolicy permanently dropped
																									 // via BatchAdmissionDecision::Reject
};

/// Arachne's index-agnostic control plane: decides where SEARCH/INSERT/DELETE
/// actually run and drives them through IAdapter/IRegion, without ever
/// depending on a concrete index or a concrete Anchor storage structure.
///
/// Two concerns are kept deliberately separate:
///  - RoutingCache (injected, pluggable): "is this query close to something
///    we've seen" -- a pure identity/routing signal.
///  - RegionManager (owned directly): which Regions an Anchor has promoted,
///    each Region's host/device/lease state, and when GPU residency actually
///    changes (see its own doc comment for why it owns the whole
///    GPU-residency policy, not just bookkeeping). Its own Coordinator
///    thread decides and performs promotion/eviction; Controller only
///    routes, schedules, and reports traversals/promotion requests into it.
///    Controller is the only class that talks to both RoutingCache and
///    RegionManager.
///
/// Dispatch flow (search() shown; insert()/remove() follow the same
/// route -> dispatch -> commit shape):
///
///   search(query)
///        |
///        v
///     route()  --- RoutingCache hit w/ promoted Regions? --> GpuOnly plan
///        |                                                   (falls back to
///        v                                                    Hybrid if it
///     dispatch(TraverseRequest)                                doesn't cover
///        |                                                      the query)
///        v
///     OpScheduler::schedule() -- batches with other same-mode requests,
///        |                       runs on an execution worker thread
///        v
///     on_complete (worker thread): RegionManager::recordTraversal() +
///        |                         requestPromotion()
///        v
///     commitSearch() -- shapes the public SearchResult
///
/// insert()/remove() additionally track which VectorIds are currently live
/// (live_ids_) so a duplicate/unknown id is rejected before it ever reaches
/// the adapter or RegionManager.
///
/// GPU access: acquireRegion() is the only sanctioned way to read a Region's
/// data; it returns a RegionAccess telling the caller whether the data is on
/// device (with a Lease pinning it) or host-only. Each OpScheduler execution
/// worker thread binds thread_local g_worker_stream (controller.cpp) to its
/// own dedicated CUDA stream once, at thread start (via scheduler_.start()'s
/// on_worker_start hook); acquireRegion() picks up that stream when called
/// from a worker thread, or DeviceContext's management stream otherwise --
/// see gpu::DeviceContext::workerStream()/managementStream()'s own doc
/// comments for why the two are kept physically separate. The same
/// on_worker_start hook also binds thread_local g_worker_scratch to that
/// worker's dedicated scratch buffer (gpu::DeviceContext::workerScratch()),
/// exposed via workerScratch() below -- see IAdapter::
/// requiredScratchBytesPerWorker()'s doc comment for what it's for.
///
/// Member destruction order matters and is enforced by declaration order
/// below: scheduler_ (whose worker threads read g_worker_stream and may
/// call back into acquireRegion()) must stop before region_manager_'s
/// Coordinator thread stops, which must stop before device_region_pool_/
/// device_ tear down.
class Controller {
 public:
	// GPU/threading knobs (gpu_*_budget_bytes, gpu_unit_bytes, compaction_policy,
	// allocation_policy) are forwarded to the DeviceContext/DeviceRegionPool this
	// Controller owns and default to those classes' own defaults; overriding
	// them is mainly useful for tests exercising capacity-exhaustion/eviction/
	// fragmentation without allocating gigabytes of real Region data first.
	// `allocation_policy` is appended last, after coordinator_config, so it
	// doesn't shift any existing positional call site.
	Controller(IAdapter& adapter, RoutingCache& routing_cache,
			 SchedulingConfig scheduling_config = {},
			 std::unique_ptr<ReplacementPolicy> replacement_policy = nullptr,
			 std::size_t gpu_data_budget_bytes = gpu::kDefaultDataPoolBytes,
			 std::size_t gpu_metadata_budget_bytes = gpu::kDefaultMetadataPoolBytes,
			 std::size_t gpu_unit_bytes = gpu::kDefaultUnitBytes,
			 std::unique_ptr<gpu::CompactionPolicy> compaction_policy = nullptr,
			 CoordinatorConfig coordinator_config = {},
			 gpu::AllocationPolicy allocation_policy = gpu::AllocationPolicy::Async);

	/// `record_for_replacement_policy = false` runs the search exactly as
	/// normal (routing, GpuOnly/Hybrid dispatch, adapter call) but suppresses
	/// both RegionManager::recordTraversal() and requestPromotion() for it --
	/// i.e. this query is invisible to promotion/eviction bookkeeping, as if
	/// it never happened, while still returning a real result. For a
	/// measurement-only query (e.g. computing recall against a held-out
	/// evaluation set) that must not itself perturb which Regions later get
	/// promoted/evicted. Defaults to true (normal behavior, matching every
	/// existing caller).
	SearchResult search(const Query& query, bool record_for_replacement_policy = true);

	/// Fails (ok=false, nothing touched) if record.id is already live -- a
	/// prior unremoved insert(), or a concurrent insert() for the same id.
	/// Enforced here since an index's own id space assumes uniqueness (e.g.
	/// StressIndex's id_to_slot_, hnswlib's label_lookup_); remove() frees
	/// the id for reuse.
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

	/// Non-blocking siblings of search()/insert()/remove(): submit the same
	/// work, but return immediately with a future instead of waiting for it
	/// to finish. search()/insert()/remove() are themselves now just
	/// `submitXxx(...).get()`.
	///
	/// Why this exists: SchedulingConfig::traverse_batch_size/
	/// modify_batch_size only let OpScheduler merge multiple *simultaneously
	/// pending* requests into one adapter call -- but search()/insert()/
	/// remove() each submit-and-immediately-block, so at most one request
	/// per calling thread is ever pending at once, capping how much any
	/// batch_size setting > (number of calling threads) can actually help.
	/// A caller that wants OpScheduler's batching to matter needs to have
	/// many requests genuinely in flight at the same time -- e.g. submit a
	/// whole step's worth of vectors via submitInsert() into a
	/// std::vector<std::future<...>> first, *then* call .get() on all of
	/// them -- which these make possible without spawning a thread per
	/// pending request.
	///
	/// Lifetime: unlike a synchronous call, the request may not actually run
	/// until well after this function returns, so (unlike VectorView's
	/// general "never assumes it outlives a single call" contract, see
	/// types.hpp) the caller must keep `record`/`query`'s backing vector
	/// memory valid until the returned future is ready -- not just until
	/// this function returns.
	///
	/// insert() is a two-stage pipeline (a Traverse lookup, then a Modify
	/// using what it found); submitInsert() chains the two internally via
	/// OpScheduler's on_complete hook (see TraverseTask::on_complete's doc
	/// comment) rather than blocking any thread on the lookup's own future,
	/// so both stages remain individually batchable across many concurrent
	/// submitInsert() calls. submitSearch() chains its own possible
	/// GpuOnly-miss-then-Hybrid-retry the same way.
	std::future<SearchResult> submitSearch(const Query& query, bool record_for_replacement_policy = true);
	std::future<InsertResult> submitInsert(const Record& record);
	std::future<DeleteResult> submitRemove(VectorId id);

	/// Adapter opt-in (see RegionManager::registerRegion()'s doc comment):
	/// declares `id` promotion/eviction-eligible and records where its host
	/// data lives. Before this call Arachne has no knowledge of `id`, and
	/// RegionManager refuses to promote any Anchor onto it.
	void registerRegion(RegionId id, HostRegionView host);

	/// Resolves where `region`'s data currently lives -- see RegionAccess's
	/// doc comment for how a caller is meant to use the result. Throws
	/// std::invalid_argument if `region` was never registered.
	RegionAccess acquireRegion(RegionId region);

	/// Persistent, worker-affine GPU scratch buffer for the calling
	/// OpScheduler execution worker thread -- see IAdapter::
	/// requiredScratchBytesPerWorker()'s doc comment for the contract (size
	/// fixed once at this Controller's construction; the same buffer is
	/// reused across every call/hop that worker thread ever makes, never
	/// individually allocated/freed). Returns nullptr if called from a
	/// non-worker thread, or if the adapter's requiredScratchBytesPerWorker()
	/// returned 0 -- a caller must treat either case as "no scratch
	/// available" and fall back to its own allocation, not as an error.
	void* workerScratch() const;

	/// The calling OpScheduler execution worker thread's own dedicated CUDA
	/// stream (gpu::DeviceContext::workerStream()) -- the same stream
	/// acquireRegion() itself picks up internally, exposed directly for a
	/// caller that needs a stream *before* knowing whether any particular
	/// Region is resident (e.g. HnswlibIndexGpu's device_query setup, which no
	/// longer depends on its entry point being resident -- see
	/// hnswlib_index_gpu.cpp). Falls back to DeviceContext's management stream
	/// when called from a non-worker thread, same as acquireRegion().
	cudaStream_t workerStream() const;

	/// See ControllerStats' own doc comment. Cheap: the counters are
	/// independent relaxed atomics, and gpu_bytes_allocated delegates to
	/// gpu::DeviceRegionPool::bytesAllocated(), which takes its own lock.
	ControllerStats stats() const;

	/// Blocks until RegionManager's Coordinator has caught up with every
	/// insert()/remove() so far -- not needed on the normal async path
	/// (promotion/eviction run off insert()/remove()'s critical path by
	/// design); for callers needing a synchronous checkpoint, e.g. a test
	/// asserting on GPU residency, or a clean point before shutdown.
	void waitIdle();

 private:
	struct SearchPlan {
		TraverseRequest primary;
		bool fallback_to_hybrid = false;
	};

	struct InsertPlan {
		ModifyRequest request;
	};

	struct RemovePlan {
		ModifyRequest request;
	};

	// Anchor Query routing (design point 1).
	struct RoutingDecision {
		bool gpu_only = false;
		RegionFootprint predicted_scope;  // derived from the matched Anchor's dependent Regions
		std::vector<RegionResidencyHint> residency_hints;
		// The matched Anchor's id, set whenever routing_cache_.nearest() hits --
		// independent of gpu_only (an adapter's entry-point cache benefits from
		// seeing this on a Hybrid request too, so it can populate itself for a
		// later GpuOnly attempt at the same Anchor). See TraverseRequest::anchor_id
		// (adapter/index_adapter.hpp) for the caveat about what this id means.
		std::optional<VectorId> anchor_id;
	};
		RoutingDecision route(const Query& query);
		SearchPlan routeSearch(const Query& query);
		// `candidates` is the lookup traversal's result (see insert()'s doc
		// comment): its `touched` folds into the resulting ModifyRequest::scope
		// (narrowed further below if a Region is already promoted), and `hint`
		// is moved rather than copied since `candidates` isn't used afterward.
		InsertPlan routeInsert(const Record& record, TraverseResult candidates);
		RemovePlan routeRemove(VectorId id);

	// Mints a fresh Anchor id, used whenever routing didn't already find an
	// existing Anchor to attach to (both search()'s and insert()'s own
	// lookup traversal need this) -- see next_anchor_id_'s own doc comment
	// for why this replaced insert()'s previous "just reuse record.id"
	// shortcut, and index_adapter.hpp's TraverseRequest::anchor_id doc
	// comment for the opaque-id contract this now upholds unconditionally
	// rather than only for the search() path. Guaranteed collision-free with
	// every id minted so far unless next_anchor_id_ has completed a full
	// 64-bit lap -- see this method's own definition (controller.cpp) for
	// how that residual case is detected and handled rather than merely
	// documented away.
	VectorId MintAnchorId();

	// `promotion_anchor_id`, when nonzero, is offered to RegionManager::
	// requestPromotion() as a promotion candidate for whatever Regions this
	// traversal touches; that call and recordTraversal() both run on the
	// worker thread via on_complete (see class doc comment), not here.
	// Defaults to 0 (no promotion request) -- e.g. verify()'s traversal.
	// `record_traversal = false` additionally suppresses recordTraversal()
	// itself (so promotion_anchor_id is moot regardless of its value) -- see
	// search()'s `record_for_replacement_policy` doc comment for why a caller
	// would want this.
	TraverseResult dispatch(const TraverseRequest& request, VectorId promotion_anchor_id = 0,
			bool record_traversal = true);
	ModifyResult dispatch(const ModifyRequest& request);

	// Non-blocking sibling dispatch() is now a thin wrapper over: schedules
	// `request`, wires up the same recordTraversal()/requestPromotion()
	// on_complete bookkeeping, but returns the future as-is instead of
	// calling .get() on it. submitSearch()/submitInsert() do *not* go
	// through this: their retry/chaining on_complete needs to do more than
	// just this bookkeeping (schedule a follow-up request, resolve an outer
	// promise), which a plain returned future can't have attached after the
	// fact -- they call scheduler_.schedule() directly instead, inlining the
	// same two bookkeeping lines themselves.
	std::future<TraverseResult> dispatchAsync(
			const TraverseRequest& request, VectorId promotion_anchor_id = 0, bool record_traversal = true);

	// RegionManager owns RoutingCache registration/erasure itself now, at
	// actual promotion-grant/eviction time -- these just shape each
	// primitive's public Result from dispatch()'s output, plus (commitRemove
	// only) releasing `anchor_id`'s Region dependencies on success.
	SearchResult commitSearch(const TraverseResult& result, bool final_was_hybrid);
	InsertResult commitInsert(const ModifyResult& result);
	DeleteResult commitRemove(const RemovePlan& plan, const ModifyResult& result);

	// Selective verification (design point 3). Not yet wired into search().
	// On mismatch, reclaims every Region dependency on `anchor_id` (via
	// region_manager_.releaseAnchor()) since the regions it currently points
	// at no longer represent its locality.
	void verify(const Query& query, VectorId anchor_id, const TraverseResult& gpu_only_result);

	IAdapter& adapter_;
	RoutingCache& routing_cache_;
	// GPU residency accounting is Arachne-owned, not the adapter's; region_manager_'s
	// Coordinator allocates/frees through device_region_pool_.
	gpu::DeviceContext device_;
	gpu::DeviceRegionPool device_region_pool_;
	// Declared after device_region_pool_: RegionManager's Coordinator thread
	// touches device_region_pool_ up until shutdown() joins it, so it must
	// destroy first -- reverse declaration order -- before device_region_pool_/
	// device_ tear down.
	RegionManager region_manager_;
	// Declared last: scheduler_'s worker threads must stop before device_/
	// device_region_pool_ tear down (see class doc comment for why); an
	// adapter calling acquireRegion() from traverseDevice()/modifyDevice()
	// would otherwise touch already-destroyed GPU state.
	OpScheduler scheduler_;
	// Backing counter for MintAnchorId() -- both search()'s and insert()'s own
	// lookup traversal mint from this independently (atomic, no other lock
	// protecting it) whenever routing didn't already find an existing Anchor.
	// Anchor ids are a namespace Core manages entirely on its own, distinct
	// from the adapter's own VectorId space (an Anchor minted here need not
	// correspond to any real, stored element -- see index_adapter.hpp's
	// TraverseRequest::anchor_id doc comment) -- MintAnchorId() ORs the top
	// bit onto every value this counter produces, guaranteeing that
	// separation by construction rather than by convention. This counter
	// itself is a plain, unbounded 64-bit increment -- it isn't masked to 63
	// bits, so its raw values keep climbing past 2^63 (at which point OR-ing
	// the top bit becomes a no-op, harmlessly) all the way up to the actual
	// std::uint64_t wraparound at 2^64. See MintAnchorId()'s own doc comment
	// for what happens on that wraparound (astronomically unlikely on its
	// own terms, not merely "unlikely at today's scale").
	std::atomic<VectorId> next_anchor_id_{1};
	// Sticky "has next_anchor_id_ ever completed a full lap" flag --
	// MintAnchorId() only pays isKnownAnchor()'s mutex-guarded collision
	// check once this is true, keeping every mint before the first (if
	// ever) wraparound completely lock-free on this member. See
	// MintAnchorId()'s own doc comment for the full reasoning.
	std::atomic<bool> next_anchor_id_wrapped_{false};

	// Which VectorIds insert() currently considers live (see its doc
	// comment) -- independent of region_manager_, since an id can be live
	// without ever having a promoted Region, so regionsOf(id) being empty
	// isn't a reliable "was this id ever inserted" signal.
	std::mutex live_ids_mutex_;
	std::unordered_set<VectorId> live_ids_;
};

}  // namespace arachne
