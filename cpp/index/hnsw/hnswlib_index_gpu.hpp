#pragma once

// HnswlibIndexGpu: traverseDevice() offloads per-hop candidate distance
// computation to a GPU kernel (hnsw_dist_kernel.cu/.cuh); control flow
// (candidate queue, visited set, stopping condition) stays entirely on
// host -- report.md §7.3/§9's "shallow offload" Phase 2 design. Exactly one
// candidate is expanded per round-trip by default (BeamWidth() below
// defaults to 1) -- widening it trades more host<->GPU round-trips saved
// against computing some distances a strict best-first walk wouldn't have
// needed yet; not tuned/benchmarked here, kept as an internal knob rather
// than exposed.
//
// Genuine multi-query batching (TraverseBatchOnDevice(), hnswlib_index_gpu.cpp):
// traverseDevice() receives OpScheduler's whole ScheduledOperationBatch as
// one std::vector<TraverseRequest> already (op_scheduler.cpp's
// executeTraverseBatch() hands it over in a single call, never one request
// at a time) -- what used to happen here was looping that vector and
// calling a single-query search once per request, i.e. N sequential walks
// sharing nothing but a host thread. TraverseBatchOnDevice() instead runs
// every request's search *concurrently*, hop-synchronized: each request
// keeps its own independent candidate queue/top-k/visited state (exactly
// TraverseOneOnDevice()'s old per-query state, now one instance per
// request), but every round, whichever requests still have work combine
// their own frontier's new candidates into ONE list and issue ONE GPU
// kernel launch together -- see hnsw_dist_kernel.cuh's extended
// LaunchDistanceKernel() (now takes a per-candidate query index alongside
// the query batch). A request that converges early (its own best-first
// stop condition trips, or its candidate queue empties) simply stops
// contributing candidates to later rounds -- it costs nothing further, not
// even an idle GPU lane, since the combined per-round candidate list is
// built by literally omitting finished requests, not by padding a fixed
// grid shape. This is what actually amortizes the host<->device round-trip
// (the real bottleneck this whole design targets, see report.md §5's
// SVFusion lesson) across a batch, instead of just amortizing host-side
// call overhead the way handing the vector over in one call already did at
// the OpScheduler boundary. A batch of exactly one request is the
// degenerate case: every round has exactly one contributor, so the trace
// is bit-for-bit identical to the old single-query loop.
//
// Entry point (this class's second responsibility, originally a separate
// sibling class -- HnswIndexAnchorEntry -- until it was merged in here, see
// git history and report.md §10.4 for the design discussion that led to the
// merge): resolveEntryPoint() below is overridden to opportunistically skip
// hnswlib's own global entry point by consulting anchor_entry_point_, a
// cache of Anchor id -> the internal id a prior *completed* traverseHost()
// call for that Anchor actually landed on (populated by this class's own
// traverseHost() override as a side effect). A cache miss (no anchor_id, or
// this Anchor's first-ever GpuOnly attempt) falls back to HnswlibIndex's
// plain global-entry-point default -- so this is a strict improvement over
// the un-cached behavior, never a regression: traverseDevice() itself never
// populates the cache, only consumes it, so a Region's first-ever GpuOnly
// walk is always exactly as good as before the merge, and only a *repeat*
// visit to the same locality benefits.
//
// hnswlib source is NOT reused for the search loop here, unlike the rest
// of this directory -- hnswlib's own searchBaseLayerST() (thirdparty/
// hnswlib/hnswlib/hnswalg.h) calls its private fstdistfunc_ inline, with
// no seam to substitute a GPU-batched distance step without either
// patching hnswlib itself (rejected, see hnswlib_index.hpp's "used as-is"
// guarantee) or reimplementing the loop. This class does the latter --
// TraverseBatchOnDevice() (hnswlib_index_gpu.cpp) is a from-scratch
// re-implementation of hnswlib's bare-bone level-0 greedy search
// (candidate priority queue + bounded top-k + visited set), built
// entirely on top of HnswlibIndex's already-public/protected surface
// (resolveEntryPoint(), engineLevel0Neighbors(), engineDataPointerFor(),
// engineIsMarkedDeleted(), engineExternalLabel(), resolveRegion()) rather
// than touching hnswlib internals directly a second time. Unlike hnswlib's
// own searchKnn(), this never descends through levels above 0 -- it starts
// directly at resolveEntryPoint()'s result and walks level 0 only (see
// report-hnsw-dist.md for the recall/hop-count implications of that gap).
//
// GPU residency, explicitly (report.md's own correction of "offload = GPU
// parallelism"): every candidate a round needs is resolved through
// Controller::acquireRegion() before its distance is computed. Originally
// (see git history), any single non-resident candidate aborted the *whole*
// call -- TraverseResult::completed_within_scope=false, discarding every
// hop already computed, with Controller's fallback_to_hybrid machinery (see
// core/controller.hpp's routeSearch()) left to retry the *entire* query from
// scratch via HnswlibIndex::traverseHost(). TraverseBatchOnDevice() (hnswlib_
// index_gpu.cpp) no longer does that: compute_distances_batch() resolves
// residency per *candidate*, batches whichever subset of a round is GPU-resident into one
// kernel launch as before, and computes the rest directly on host via
// engineHostDistance() (hnswlib's own fstdistfunc_, called through
// HnswEngine -- see that method's own doc comment for why this is safe to
// mix with the GPU path and still match traverseHost() exactly). The walk
// itself never aborts for residency reasons anymore -- completed_within_
// scope is unconditionally true -- so "GpuOnly" for this class now means
// "GPU whenever the data is there, host per candidate otherwise," not "GPU
// or nothing." Level-0 adjacency/deletion/label lookups were already
// unconditional host reads regardless (see engineLevel0Neighbors()/
// engineIsMarkedDeleted()/engineExternalLabel()'s own doc comments) and stay
// that way -- only the vector *values* used for distance computation were
// ever residency-gated, and now even those degrade per-candidate instead of
// all-or-nothing.
//
// TraverseResult::touched is populated too (it wasn't before): every
// internal id whose distance gets computed this walk, GPU or host, not just
// the final top-k -- see TraverseBatchOnDevice()'s own comment for why this is
// deliberately richer than HnswlibIndex::traverseHost()'s top-k-only
// approximation. This is what feeds RegionManager's promotion/eviction
// hotness signal (Controller::search()'s on_complete -> recordTraversal())
// for a device-served query -- without it, GPU-served traffic contributed no
// hotness signal at all, regardless of how much of the graph it actually
// touched.
//
// Every (VectorDType, DistanceMetric) combination hnswlib itself supports
// is covered on the device path except Cosine (throws std::logic_error for
// Cosine -- hnswlib has no native Cosine Space, same reason the host path's
// makeEngine() rejects it, see hnswlib_index.cpp). hnsw_dist_kernel.cu's own
// overview documents which hnswlib scalar distance function each
// (dtype, metric) pair's kernel matches bit-for-bit (L2Sqr/L2SqrI/
// L2SqrInt8/L2SqrHalf and InnerProduct/InnerProductU8/InnerProductInt8/
// InnerProductHalf's scalar fallback paths).

#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "hnswlib_index.hpp"

namespace arachne {
class Controller;
}

namespace arachne::index::hnsw {

class HnswlibIndexGpu final : public HnswlibIndex {
 public:
	/// `max_batch_size` sizes requiredScratchBytesPerWorker()'s reservation
	/// (see that method's own doc comment) -- it should match (or exceed)
	/// whatever SchedulingConfig::traverse_batch_size the caller configures
	/// its OpScheduler with, since that's the real upper bound on how many
	/// requests one traverseDevice() call ever carries. Nothing breaks if it
	/// doesn't: TraverseBatchOnDevice() falls back to a one-off cudaMalloc for
	/// whichever buffer a call exceeds reserved capacity for (same fallback
	/// this class already had per-round for an oversized candidate count,
	/// see hnswlib_index_gpu.cpp's ComputeScratchLayout()) -- a mismatch only
	/// costs the fast path for the calls that exceed it, never correctness.
	/// Defaults to 1, i.e. the same scratch footprint as before this
	/// constructor parameter existed.
	HnswlibIndexGpu(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
									 std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction,
									 std::size_t max_batch_size = 1);
	~HnswlibIndexGpu() override = default;

	/// Must be called once, after registerAllRegions(controller) against the
	/// same `controller`. Not part of IAdapter/HnswlibIndex's own interface
	/// (mirrors why registerAllRegions() itself isn't part of IAdapter
	/// either): traverseDevice() needs to call back into
	/// Controller::acquireRegion() (report.md §9.3's "host acquires Leases
	/// right before kernel launch" design), and IAdapter::traverseDevice()'s
	/// fixed signature has no way to pass a Controller reference in per-call.
	/// traverseDevice() throws std::logic_error if this was never called.
	void attachController(Controller& controller) { controller_ = &controller; }

	std::vector<TraverseResult> traverseDevice(const std::vector<TraverseRequest>& requests) override;

	/// Identical results to HnswlibIndex::traverseHost() -- hnswlib's own
	/// searchKnnCloserFirst() has no seam to consult resolveEntryPoint(), so
	/// the search itself is untouched; overridden only to record, for each
	/// request carrying an anchor_id and returning at least one neighbor,
	/// that Anchor's entry-point cache entry (anchor_entry_point_ below) --
	/// see resolveEntryPoint() for how it's consumed. anchor_entry_point_ is
	/// unbounded and never evicted (an Anchor id that's released on the
	/// Controller side has no corresponding notification here) -- acceptable
	/// for now, not solved here.
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override;

	/// See IAdapter's own doc comment. Sized to hold, for the current
	/// dtype()/dim()/BeamWidth()/M()/max_batch_size (ctor parameter above):
	/// up to max_batch_size query vectors, plus one hop-synchronized round's
	/// worth of combined candidate device-pointers, query-index tags, and
	/// distance outputs at the worst case (max_batch_size requests each
	/// contributing BeamWidth() candidates, each expanding up to hnswlib's
	/// level-0 max degree, M()*2) -- see hnswlib_index_gpu.cpp's
	/// ComputeScratchLayout() for the exact arithmetic, shared with
	/// TraverseBatchOnDevice()'s actual buffer use so the two can never drift
	/// apart. A call/round that (rarely) exceeds this falls back to a
	/// one-off cudaMalloc/cudaFree for just the buffer(s) that didn't fit --
	/// see TraverseBatchOnDevice()'s own comment.
	std::size_t requiredScratchBytesPerWorker() const override;

 protected:
	/// On a cache hit for request.anchor_id, returns the internal id a prior
	/// *completed* traverseHost() call for that Anchor actually landed on
	/// (see traverseHost() above for how it gets there). Falls back to
	/// HnswlibIndex's own global-entry-point default otherwise (no
	/// anchor_id, or not cached yet -- e.g. this Anchor's first-ever GpuOnly
	/// attempt).
	std::uint32_t resolveEntryPoint(const TraverseRequest& request) const override;

	/// How many candidates TraverseBatchOnDevice() pops off each request's own
	/// `candidate_set` (and batches into that round's combined GPU distance
	/// call, alongside every other still-active request's own frontier) per
	/// round, instead of exactly one. Widening this trades fewer host<->GPU
	/// round-trips against computing some distances a strict best-first walk
	/// wouldn't have needed yet -- not tuned/benchmarked, kept at the
	/// conservative default. Widening the beam changes nothing about which
	/// candidates are *eligible* to expand -- the per-round stop condition is
	/// unchanged -- only how many of the eligible ones are taken together
	/// before the next GPU round-trip.
	std::size_t BeamWidth() const { return 1; }

 private:
	std::vector<TraverseResult> TraverseBatchOnDevice(const std::vector<TraverseRequest>& requests);

	Controller* controller_ = nullptr;
	std::size_t max_batch_size_;

	// Guards anchor_entry_point_ only -- deliberately separate from the
	// inherited mutex_ (HnswlibIndex's own doc comment), which by this point
	// guards only build()/exportTo()/loadFrom()/liveCount() and is never held
	// during traverseHost()/TraverseBatchOnDevice() at all. Kept as its own
	// dedicated mutex anyway: this map is genuinely independent state with
	// its own lifetime/contention profile, unrelated to hnswlib engine
	// access -- reusing mutex_ for it would just be an arbitrary coupling.
	mutable std::mutex anchor_cache_mutex_;
	std::unordered_map<VectorId, std::uint32_t> anchor_entry_point_;
};

}  // namespace arachne::index::hnsw
