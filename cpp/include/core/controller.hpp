#pragma once

#include <atomic>
#include <optional>
#include <cstdint>
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
/// comments for why the two are kept physically separate.
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

	SearchResult search(const Query& query);

	/// Fails (ok=false, nothing touched) if record.id is already live -- a
	/// prior unremoved insert(), or a concurrent insert() for the same id.
	/// Enforced here since an index's own id space assumes uniqueness (e.g.
	/// StressIndex's id_to_slot_, hnswlib's label_lookup_); remove() frees
	/// the id for reuse.
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

	/// Adapter opt-in (see RegionManager::registerRegion()'s doc comment):
	/// declares `id` promotion/eviction-eligible and records where its host
	/// data lives. Before this call Arachne has no knowledge of `id`, and
	/// RegionManager refuses to promote any Anchor onto it.
	void registerRegion(RegionId id, HostRegionView host);

	/// Resolves where `region`'s data currently lives -- see RegionAccess's
	/// doc comment for how a caller is meant to use the result. Throws
	/// std::invalid_argument if `region` was never registered.
	RegionAccess acquireRegion(RegionId region);

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
	};
		RoutingDecision route(const Query& query);
		SearchPlan routeSearch(const Query& query);
		// `candidates` is the lookup traversal's result (see insert()'s doc
		// comment): its `touched` folds into the resulting ModifyRequest::scope
		// (narrowed further below if a Region is already promoted), and `hint`
		// is moved rather than copied since `candidates` isn't used afterward.
		InsertPlan routeInsert(const Record& record, TraverseResult candidates);
		RemovePlan routeRemove(VectorId id);

	// `promotion_anchor_id`, when nonzero, is offered to RegionManager::
	// requestPromotion() as a promotion candidate for whatever Regions this
	// traversal touches; that call and recordTraversal() both run on the
	// worker thread via on_complete (see class doc comment), not here.
	// Defaults to 0 (no promotion request) -- e.g. verify()'s traversal.
	TraverseResult dispatch(const TraverseRequest& request, VectorId promotion_anchor_id = 0);
	ModifyResult dispatch(const ModifyRequest& request);

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
	// Minted from search()'s calling thread whenever a query needs a Hybrid
	// traversal -- atomic since multiple concurrent search() calls mint from
	// this independently, with no other lock protecting it.
	std::atomic<VectorId> next_anchor_id_{1};

	// Which VectorIds insert() currently considers live (see its doc
	// comment) -- independent of region_manager_, since an id can be live
	// without ever having a promoted Region, so regionsOf(id) being empty
	// isn't a reliable "was this id ever inserted" signal.
	std::mutex live_ids_mutex_;
	std::unordered_set<VectorId> live_ids_;
};

}  // namespace arachne
