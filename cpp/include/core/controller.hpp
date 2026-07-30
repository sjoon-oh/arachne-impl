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

/// Arachne's core management controller: the index-agnostic control plane that
/// decides where SEARCH/INSERT/DELETE run, per the Quick Summary. This is
/// the class meant to carry Arachne's actual design as it gets built out;
/// everything here is implemented against IAdapter/IRegion and RoutingCache
/// only, never against a concrete index or a concrete Anchor storage
/// structure.
///
/// Splits "is this query close to something we've seen" (RoutingCache,
/// injected -- pure identity/routing signal, pluggable) from "which Regions
/// has that Anchor promoted, and what does each Region's own
/// host/device/lease state look like, and when does GPU residency actually
/// change" (RegionManager, owned directly -- see its own doc comment for why
/// it now owns the whole GPU-residency policy, not just bookkeeping).
/// Controller routes and schedules; RegionManager's own Coordinator decides
/// and performs promotion/eviction. Controller is the only thing that talks
/// to both RoutingCache and RegionManager.
class Controller {
 public:
	// `replacement_policy` is forwarded to region_manager_ (which owns it --
	// see RegionManager's own doc comment), defaulting to
	// FifoReplacementPolicy when left null, mirroring OpScheduler's own
	// SchedulingPolicy default. `gpu_data_budget_bytes`/
	// `gpu_metadata_budget_bytes` become the DeviceContext this Controller
	// owns' budgetBytes() for each MemoryKind -- the ceiling
	// RegionManager::make()'s gpu::DeviceRegionPool::tryAllocate() calls
	// enforce (see its own doc comment). Defaulted to the same
	// kDefaultDataPoolBytes/kDefaultMetadataPoolBytes DeviceContext itself
	// defaults to; overriding them is mainly useful for tests that want to
	// exercise capacity-exhaustion/eviction without allocating gigabytes of
	// real Region data first.
	Controller(IAdapter& adapter, RoutingCache& routing_cache,
			 SchedulingConfig scheduling_config = {},
			 std::unique_ptr<ReplacementPolicy> replacement_policy = nullptr,
			 std::size_t gpu_data_budget_bytes = gpu::kDefaultDataPoolBytes,
			 std::size_t gpu_metadata_budget_bytes = gpu::kDefaultMetadataPoolBytes,
			 CoordinatorConfig coordinator_config = {});

	SearchResult search(const Query& query);

	/// Fails (InsertResult::ok = false, no adapter/RoutingCache/RegionManager
	/// call made at all) if record.id is already live -- either a prior
	/// insert() for the same id already succeeded and it hasn't been
	/// remove()'d since, or another thread's insert() for the same id is
	/// concurrently in flight right now. An index's own id space assumes
	/// uniqueness (e.g. StressIndex's id_to_slot_ map, or hnswlib's
	/// label_lookup_); Arachne enforces that at this boundary rather than
	/// letting a duplicate reach the adapter and silently corrupt whatever
	/// id->data mapping it keeps. remove()'ing an id frees it for reuse.
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

	/// Adapter opt-in (see RegionManager::registerRegion()'s doc comment):
	/// declares `id` promotion/eviction-eligible and records where its data
	/// lives in host memory. An index calls this once it decides a piece of
	/// its own state should be a candidate for GPU residency -- before that
	/// call, Arachne has no knowledge of `id` at all, and RegionManager will
	/// refuse to promote any Anchor onto it.
	void registerRegion(RegionId id, HostRegionView host);

	/// Resolves where `region`'s data currently lives -- see RegionAccess's
	/// doc comment for how a caller is meant to use the result. Throws
	/// std::invalid_argument if `region` was never registered.
	RegionAccess acquireRegion(RegionId region);

	/// See ControllerStats' own doc comment. Cheap: the four counters are
	/// independent atomics read with relaxed ordering (nothing here needs to
	/// be seen-together-atomically with anything else -- each is just a
	/// running total), and gpu_bytes_allocated delegates to
	/// gpu::DeviceRegionPool::bytesAllocated(), which takes its own lock.
	ControllerStats stats() const;

	/// Blocks until RegionManager's Coordinator has fully caught up with
	/// every insert()/remove() call made so far -- see
	/// RegionManager::waitIdle()'s own doc comment. Not needed on the normal
	/// async path (promotion/eviction happen off of insert()/remove()'s own
	/// critical path by design); exists for callers that need a synchronous
	/// checkpoint -- e.g. a test asserting on GPU residency right after a
	/// batch of inserts, or an operator wanting a clean point before
	/// shutdown.
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
	};
		RoutingDecision route(const Query& query);
		SearchPlan routeSearch(const Query& query);
		// `candidates` is the result of the lookup traversal insert() runs
		// first (see its doc comment) -- routeInsert() folds `candidates.touched`
		// into the resulting ModifyRequest::scope (which may later be narrowed
		// to a single already-promoted Region -- see the loop below), and moves
		// (not copies) `candidates.hint` into ModifyRequest::hint, since
		// `candidates` isn't needed for anything else afterward. Promotion
		// itself is already requested by the time this runs -- see dispatch()'s
		// own doc comment.
		InsertPlan routeInsert(const Record& record, TraverseResult candidates);
		RemovePlan routeRemove(VectorId id);

	// `promotion_anchor_id`, when nonzero, requests replacement_policy_
	// consider `promotion_anchor_id` a promotion candidate for whatever
	// Regions this traversal touches -- see RegionManager::requestPromotion().
	// Both this and the recordTraversal() hotness signal run on the
	// OpScheduler execution worker thread that actually computed the result
	// (via OpScheduler::schedule()'s on_complete hook), not on whichever
	// thread calls this and blocks on the future -- see that hook's own doc
	// comment for why: it keeps contention on RegionManager's lock bounded by
	// the (small, fixed) worker pool instead of by however many
	// search()/insert() callers happen to be concurrently in flight. The same
	// callback also passes `request.query.vector` through to
	// requestPromotion() -- safe to capture here (before the worker thread
	// even runs) because the original caller is still blocked on this call's
	// own future.get() below, keeping the caller-owned buffer alive for the
	// callback's duration; see PromotionCandidate's own doc comment
	// (replacement_policy.hpp) for why RegionManager needs its own copy
	// beyond that point. Defaults to 0 (no promotion request) -- e.g.
	// verify()'s traversal.
	TraverseResult dispatch(const TraverseRequest& request, VectorId promotion_anchor_id = 0);
	ModifyResult dispatch(const ModifyRequest& request);

	// RegionManager now owns RoutingCache registration/erasure itself, at
	// actual promotion-grant/eviction time (see its own class doc comment) --
	// commitSearch()/commitInsert()/commitRemove() no longer touch
	// routing_cache_ at all. What's left of the "commit" step is just
	// shaping each primitive's final public Result type from what dispatch()
	// returned (plus, for commitRemove(), the release of `anchor_id`'s own
	// Region dependencies, which stays here since it's keyed off the Modify
	// primitive's own success, not the lookup traversal's).
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
	// GPU residency accounting (design point 4): Arachne-owned, not the
	// adapter's -- see gpu/device_context.hpp and gpu/device_region_pool.hpp.
	// region_manager_'s Coordinator allocates/frees through
	// device_region_pool_ -- see RegionManager's own doc comment for why it,
	// not Controller, now owns that whole policy.
	gpu::DeviceContext device_;
	gpu::DeviceRegionPool device_region_pool_;
	// Declared after device_region_pool_ deliberately: RegionManager's
	// Coordinator thread (started in the constructor body, stopped in
	// RegionManager's own destructor) touches device_region_pool_ up until
	// shutdown() joins it, so it must be destroyed -- reverse declaration
	// order -- *before* device_region_pool_/device_ are torn down.
	RegionManager region_manager_;
	// Declared last of the threading members deliberately: members destroy
	// in reverse declaration order, so scheduler_ -- and the execution
	// worker threads it owns -- must stop *before* device_/device_region_pool_
	// tear down (a worker thread's thread_local g_worker_stream, set from
	// gpu::DeviceContext::workerStream(), would otherwise dangle for
	// however long the worker takes to notice shutdown() and stop, and any
	// adapter that calls Controller::acquireRegion() from within
	// traverseDevice()/modifyDevice() -- unlike the adapters in this
	// codebase's own tests today -- would be touching already-destroyed GPU
	// state).
	OpScheduler scheduler_;
	// Minted from search()'s own calling thread whenever a query needs a
	// Hybrid traversal (see search()'s doc comment) -- atomic because
	// multiple concurrent search() calls mint from this independently, with
	// no other lock protecting it.
	std::atomic<VectorId> next_anchor_id_{1};

	// Which VectorIds insert() currently considers live -- see insert()'s
	// own doc comment. A plain mutex-guarded set, independent of
	// region_manager_: an id can be live (successfully handed to the
	// adapter's modifyHost()/modifyDevice()) without ever having a promoted
	// Region (promotion can fail or simply not be attempted), so
	// region_manager_.regionsOf(id) being empty is not a reliable "was this
	// id ever inserted" signal.
	std::mutex live_ids_mutex_;
	std::unordered_set<VectorId> live_ids_;
};

}  // namespace arachne
