#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "adapter/region.hpp"
#include "types.hpp"

namespace arachne {

enum class ExecutionMode {
	GpuOnly,  // must complete entirely within the given scope's resident regions
	Hybrid,   // the underlying index may fall back to Host traversal
};

/// Generic, index-defined payload Core shuttles between calls without ever
/// interpreting it -- the same principle as adapter/region.hpp's
/// ModificationDelta, applied one step earlier in the pipeline: what a
/// traversal found that should guide a following modification (graph
/// neighbor candidates for HNSW-style insertion, a cluster assignment for
/// an IVF-style index, or anything else a future index needs) is entirely
/// index-specific, and the same adapter implements both the traversal that
/// produces it (TraverseResult::hint) and the modification that consumes it
/// (ModifyRequest::hint) -- so only that adapter needs to agree with itself
/// on the encoding. Deliberately not a fixed shape like a neighbor list:
/// that would bake in a graph-shaped answer for every index, including ones
/// (cluster-based, ...) it doesn't fit.
struct OpaqueData {
	std::vector<std::byte> payload;
};

struct TraverseRequest {
	Query query;
	ExecutionMode mode = ExecutionMode::Hybrid;
	RegionFootprint scope;  // regions the traversal is confined to when GpuOnly
};

struct TraverseResult {
	SearchResult result;
	RegionFootprint touched;               // footprint actually accessed
	bool completed_within_scope = false;   // false => a GpuOnly attempt fell short
	OpaqueData hint;  // see OpaqueData's doc comment; populated at the adapter's own discretion
};

enum class ModifyOp { Insert, Delete };

struct ModifyRequest {
	ModifyOp op = ModifyOp::Insert;
	Record record;           // valid for Insert
	VectorId target = 0;     // valid for Delete
	ExecutionMode mode = ExecutionMode::Hybrid;
	RegionFootprint scope;
	LeaseHandle lease;  // set when a GPU write lease already covers `scope`
	/// For Insert: moved through (see Controller::insert()) from the
	/// TraverseResult::hint of the lookup traversal Core runs first to find
	/// candidate placement info. Default-constructed (empty) for Delete --
	/// deletion is targeted by VectorId alone, with no preceding traversal
	/// to source a hint from (see Controller::remove()'s doc comment).
	OpaqueData hint;
};

struct ModifyResult {
	bool ok = false;
	RegionFootprint touched;   // regions the candidate search/traversal read
	RegionFootprint modified;  // regions actually mutated
};

/// The only surface an underlying ANNS index must implement to be driven by
/// Arachne. Per Quick Summary: SEARCH = Traversal, INSERT = Traversal ->
/// Modification, DELETE = Modification. Arachne's Core owns *where*
/// traverse/modify run (Host/GPU/hybrid, which regions); the index owns
/// *how*. Implementing this (plus IRegion) is the integration point for a
/// concrete index -- left for a future, separate piece of work.
///
/// Thread-safety: OpScheduler (core/op_scheduler.hpp) may call
/// traverseHost()/modifyHost()/traverseDevice()/modifyDevice() concurrently
/// from as many worker threads as SchedulingConfig::max_execution_threads
/// allows -- an implementation must be safe under that (a straight wrapper
/// around an index that already handles its own internal concurrency, like
/// hnswlib's HierarchicalNSW with its per-node
/// link_list_locks_/label_lookup_lock, is generally fine as-is; one that
/// isn't needs its own locking, or the caller must be configured with
/// max_execution_threads = 1).
class IndexAdapter {
 public:
	virtual ~IndexAdapter() = default;

	/// Host-orchestrated entry point: the index is free to do its own walk
	/// here, touching GPU-resident data incrementally (e.g. per graph hop,
	/// "offloading" a sub-computation -- see ExecutionMode::Hybrid) via
	/// whatever handle Core hands it to reach promoted Regions. This is the
	/// natural shape for an index whose own traversal is inherently
	/// sequential/data-dependent (each step's target depends on the previous
	/// step's result, as in HNSW's greedy graph walk) -- see
	/// traverseDevice()'s doc comment for the alternative shape.
	///
	/// Takes a *vector* of requests -- not framed as "the single-request
	/// case" -- because even a host-only index can often batch its own
	/// CPU-side work (e.g. SIMD distance computation across several queries
	/// at once); this isn't only for GPU-native batching. A caller wanting
	/// one request handled in isolation just passes a vector of size 1.
	/// Every adapter must implement this: it's the baseline, GPU-independent
	/// path.
	///
	/// Must return exactly one result per request, in the same order as
	/// `requests` -- OpScheduler (see executeTraverseBatch()) matches
	/// results back to callers positionally and treats a mismatched count as
	/// every request in the batch having failed.
	virtual std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) = 0;
	virtual std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) = 0;

	/// Device-native entry point: every request in `requests` is handed to
	/// the index together, to run natively on GPU -- e.g. a single kernel
	/// launch (or a handful of them) spanning the whole batch, in the style
	/// of RAFT's own CAGRA. This only pays off when the index's own
	/// algorithm processes the batch hop-synchronized (every request's step
	/// k computed together, then every request's step k+1, ...); an
	/// override that just loops calling its own host logic internally gets
	/// no benefit over traverseHost()/modifyHost() and shouldn't bother
	/// overriding this at all.
	///
	/// Default implementation throws std::logic_error. Deliberately *not* a
	/// silent fallback to traverseHost()/modifyHost(): Controller only ever
	/// routes a request here when RegionManager reports the region is
	/// genuinely GPU-promoted (see ExecutionMode::GpuOnly), so reaching this
	/// default means either an adapter promoted a Region it can't actually
	/// serve from GPU, or forgot to override this -- both are bugs that
	/// should fail loudly during development, not silently degrade into
	/// running on the host every time (which would defeat the entire point
	/// of having promoted the Region, and is exactly the kind of silent
	/// GpuOnly -> host fallback Controller itself was changed *not* to do).
	virtual std::vector<TraverseResult> traverseDevice(const std::vector<TraverseRequest>& requests);
	virtual std::vector<ModifyResult> modifyDevice(const std::vector<ModifyRequest>& requests);

	/// Structural accessors, not primitives: let Core resolve footprints
	/// returned above into IRegion callbacks for lease management.
	virtual IRegion* resolveRegion(RegionId id) = 0;
	virtual std::vector<RegionId> allRegions() const = 0;
};

inline std::vector<TraverseResult> IndexAdapter::traverseDevice(const std::vector<TraverseRequest>&) {
	throw std::logic_error(
			"IndexAdapter::traverseDevice: not implemented by this adapter (a GpuOnly request was "
			"routed to an adapter with no device-native traversal)");
}

inline std::vector<ModifyResult> IndexAdapter::modifyDevice(const std::vector<ModifyRequest>&) {
	throw std::logic_error(
			"IndexAdapter::modifyDevice: not implemented by this adapter (a GpuOnly request was "
			"routed to an adapter with no device-native modification)");
}

}  // namespace arachne
