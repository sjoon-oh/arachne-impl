#pragma once

// HnswIndexDist: traverseDevice() offloads per-hop candidate distance
// computation to a GPU kernel (hnsw_dist_kernel.cu/.cuh); control flow
// (candidate queue, visited set, stopping condition) stays entirely on
// host -- report.md §7.3/§9's "shallow offload" Phase 2 design. This is
// the *naive* variant: always hnswlib's own global entry point
// (HnswIndex::resolveEntryPoint()'s default), and exactly one candidate
// expanded per round-trip (BeamWidth() below defaults to 1). Kept as its
// own class -- not a runtime toggle -- so it stays usable standalone as a
// baseline to measure HnswIndexAnchorEntry (hnsw_index_anchor_entry.hpp)
// against: same GPU-offload plumbing (CUDA calls, Controller::acquireRegion()
// residency checks, dirty-header-aware device addressing), reused via
// inheritance rather than duplicated, with only the entry point and beam
// width hooks below overridden.
//
// hnswlib source is NOT reused for the search loop here, unlike the rest
// of this directory -- hnswlib's own searchBaseLayerST() (thirdparty/
// hnswlib/hnswlib/hnswalg.h) calls its private fstdistfunc_ inline, with
// no seam to substitute a GPU-batched distance step without either
// patching hnswlib itself (rejected, see hnsw_index.hpp's "used as-is"
// guarantee) or reimplementing the loop. This class does the latter --
// TraverseOneOnDevice() (hnsw_index_dist.cpp) is a from-scratch
// re-implementation of hnswlib's bare-bone level-0 greedy search
// (candidate priority queue + bounded top-k + visited set), built
// entirely on top of HnswIndex's already-public/protected surface
// (resolveEntryPoint(), engineLevel0Neighbors(), engineDataPointerFor(),
// engineIsMarkedDeleted(), engineExternalLabel(), resolveRegion()) rather
// than touching hnswlib internals directly a second time. Unlike hnswlib's
// own searchKnn(), this never descends through levels above 0 -- it starts
// directly at resolveEntryPoint()'s result and walks level 0 only (see
// report-hnsw-dist.md for the recall/hop-count implications of that gap).
//
// GPU residency, explicitly (report.md's own correction of "offload = GPU
// parallelism"): every candidate a round needs is resolved through
// Controller::acquireRegion() before its distance is computed. If any of
// them isn't GPU-resident, the whole call bails with
// TraverseResult::completed_within_scope=false and whatever top-k it had
// found so far -- Controller's own existing fallback_to_hybrid machinery
// (see core/controller.hpp's routeSearch()) is what retries the query as
// Hybrid; this class never itself reaches into host memory to keep going
// for the vector data a distance needs (level-0 adjacency/deletion/label
// lookups are still read from host unconditionally -- see
// engineLevel0Neighbors()/engineIsMarkedDeleted()/engineExternalLabel()'s
// own doc comments; only the vector *values* used for distance computation
// are residency-gated).
//
// Every (VectorDType, DistanceMetric) combination hnswlib itself supports
// is covered on the device path except Cosine (throws std::logic_error for
// Cosine -- hnswlib has no native Cosine Space, same reason the host path's
// makeEngine() rejects it, see hnsw_index.cpp). hnsw_dist_kernel.cu's own
// overview documents which hnswlib scalar distance function each
// (dtype, metric) pair's kernel matches bit-for-bit (L2Sqr/L2SqrI/
// L2SqrInt8/L2SqrHalf and InnerProduct/InnerProductU8/InnerProductInt8/
// InnerProductHalf's scalar fallback paths).

#include <cstddef>

#include "hnsw_index.hpp"

namespace arachne {
class Controller;
}

namespace arachne::index::hnsw {

class HnswIndexDist : public HnswIndex {
 public:
	using HnswIndex::HnswIndex;
	~HnswIndexDist() override = default;

	/// Must be called once, after registerAllRegions(controller) against the
	/// same `controller`. Not part of IAdapter/HnswIndex's own interface
	/// (mirrors why registerAllRegions() itself isn't part of IAdapter
	/// either): traverseDevice() needs to call back into
	/// Controller::acquireRegion() (report.md §9.3's "host acquires Leases
	/// right before kernel launch" design), and IAdapter::traverseDevice()'s
	/// fixed signature has no way to pass a Controller reference in per-call.
	/// traverseDevice() throws std::logic_error if this was never called.
	void attachController(Controller& controller) { controller_ = &controller; }

	std::vector<TraverseResult> traverseDevice(const std::vector<TraverseRequest>& requests) override;

 protected:
	/// How many candidates TraverseOneOnDevice() pops off `candidate_set` (and
	/// batches into one GPU distance call) per round, instead of exactly one.
	/// B=1 (this class's default) is the exact original single-candidate-per-
	/// round-trip behavior. HnswIndexAnchorEntry overrides this to widen the
	/// per-round batch -- see its own file for why (fewer host<->GPU
	/// round-trips at the cost of computing some distances a strict
	/// best-first walk wouldn't have needed yet). Widening the beam changes
	/// nothing about which candidates are *eligible* to expand -- the
	/// per-round stop condition is unchanged -- only how many of the eligible
	/// ones are taken together before the next GPU round-trip.
	virtual std::size_t BeamWidth() const { return 1; }

 private:
	TraverseResult TraverseOneOnDevice(const TraverseRequest& request);

	Controller* controller_ = nullptr;
};

}  // namespace arachne::index::hnsw
