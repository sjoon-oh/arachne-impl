#include "hnswlib_index_gpu.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/controller.hpp"
#include "gpu/device_region_pool.hpp"
#include "gpu/dirty_header.hpp"
#include "hnsw_dist_kernel.cuh"
#include "logging.hpp"
#include "types.hpp"

// Implementation of HnswlibIndexGpu::traverseDevice() -- see hnswlib_index_gpu.hpp
// for the full design overview (why this reimplements the search loop
// instead of reusing hnswlib's, the residency/completed_within_scope
// contract, and the dtype/metric scope -- every (VectorDType, DistanceMetric)
// combination hnswlib itself supports except Cosine, see
// hnsw_dist_kernel.cuh's own overview).

namespace arachne::index::hnsw {

namespace {

void CheckCuda(cudaError_t status, const char* what) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("HnswlibIndexGpu: ") + what + ": " + cudaGetErrorString(status));
	}
}

using Candidate = std::pair<float, std::uint32_t>;  // (squared L2 distance, internal id)

struct MinFirst {
	bool operator()(const Candidate& a, const Candidate& b) const { return a.first > b.first; }
};

// Maps this adapter's own VectorDType/DistanceMetric onto
// hnsw_dist_kernel.cuh's self-contained KernelElemType/KernelDistanceOp --
// see that header's own comment for why it can't just use VectorDType/
// DistanceMetric directly (an nvcc/glibc header incompatibility in this
// environment when those headers' transitive includes reach the .cu file).
KernelElemType ToKernelElemType(VectorDType dtype) {
	switch (dtype) {
		case VectorDType::Float32:
			return KernelElemType::Float32;
		case VectorDType::Float16:
			return KernelElemType::Float16;
		case VectorDType::UInt8:
			return KernelElemType::UInt8;
		case VectorDType::Int8:
			return KernelElemType::Int8;
	}
	throw std::invalid_argument("HnswlibIndexGpu: unknown VectorDType");
}

KernelDistanceOp ToKernelDistanceOp(DistanceMetric metric) {
	return metric == DistanceMetric::L2 ? KernelDistanceOp::L2 : KernelDistanceOp::InnerProduct;
}

// Byte layout of one worker's persistent scratch buffer (see
// IAdapter::requiredScratchBytesPerWorker(), gpu::DeviceContext::
// workerScratch()): [queries][query_index][ptrs][out], back-to-back. Shared
// by HnswlibIndexGpu::requiredScratchBytesPerWorker() (sizing) and
// TraverseBatchOnDevice() (actual offsets) so the two can never drift apart --
// changing one without the other would either under-size the real buffer or
// silently waste memory.
struct ScratchLayout {
	std::size_t query_bytes = 0;      // query_capacity query vectors, back-to-back
	std::size_t query_capacity = 0;   // how many requests' query vectors fit in `queries` at once
	std::size_t max_candidates = 0;   // per-round cap compute_distances_batch() can serve from scratch
	std::size_t total_bytes = 0;

	void* Queries(void* base) const { return base; }
	void* QueryIndex(void* base) const { return static_cast<char*>(Queries(base)) + query_bytes; }
	void* Ptrs(void* base) const {
		return static_cast<char*>(QueryIndex(base)) + max_candidates * sizeof(std::uint32_t);
	}
	void* Out(void* base) const { return static_cast<char*>(Ptrs(base)) + max_candidates * sizeof(const void*); }
};

// query_capacity covers up to `max_batch_size` requests' query vectors at
// once (one traverseDevice() call's worth); max_candidates covers one
// hop-synchronized round's combined worst case across that whole batch:
// `max_batch_size` requests each contributing up to `beam_width` frontier
// candidates, each expanding up to hnswlib's own level-0 max degree --
// maxM0_ = M * 2 at construction time and never changed afterward
// (thirdparty/hnswlib/hnswlib/hnswalg.h), so this is a real, fixed upper
// bound given max_batch_size, not a heuristic guess. TraverseBatchOnDevice()
// still falls back to a one-off cudaMalloc for whichever buffer a call/round
// exceeds this capacity for (defensive -- expected whenever a caller's
// actual batch exceeds the max_batch_size the adapter was constructed
// with, not otherwise).
ScratchLayout ComputeScratchLayout(std::uint32_t dim, std::size_t element_bytes, std::size_t beam_width,
																		std::size_t M, std::size_t max_batch_size) {
	ScratchLayout layout;
	layout.query_capacity = std::max<std::size_t>(1, max_batch_size);
	layout.query_bytes = layout.query_capacity * std::size_t{dim} * element_bytes;
	layout.max_candidates = layout.query_capacity * beam_width * (M * 2);
	std::size_t query_index_bytes = layout.max_candidates * sizeof(std::uint32_t);
	std::size_t ptrs_bytes = layout.max_candidates * sizeof(const void*);
	std::size_t out_bytes = layout.max_candidates * sizeof(float);
	layout.total_bytes = layout.query_bytes + query_index_bytes + ptrs_bytes + out_bytes;
	return layout;
}

}  // namespace

HnswlibIndexGpu::HnswlibIndexGpu(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
																	std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction,
																	std::size_t max_batch_size)
		: HnswlibIndex(dim, dtype, metric, capacity, vectors_per_region, M, ef_construction),
			max_batch_size_(std::max<std::size_t>(1, max_batch_size)) {}

std::size_t HnswlibIndexGpu::requiredScratchBytesPerWorker() const {
	if (metric() == DistanceMetric::Cosine) return 0;  // never reaches traverseDevice() either way
	ScratchLayout layout = ComputeScratchLayout(dim(), VectorElementSize(dtype()), std::max<std::size_t>(1, BeamWidth()),
																							 M(), max_batch_size_);
	return layout.total_bytes;
}

std::vector<TraverseResult> HnswlibIndexGpu::traverseHost(const std::vector<TraverseRequest>& requests) {
	std::vector<TraverseResult> results = HnswlibIndex::traverseHost(requests);
	for (std::size_t i = 0; i < requests.size(); ++i) {
		if (!requests[i].anchor_id || results[i].result.neighbors.empty()) continue;
		VectorId top_id = results[i].result.neighbors.front().id;
		if (std::optional<std::uint32_t> internal_id = engineInternalIdFor(top_id)) {
			std::lock_guard<std::mutex> lock(anchor_cache_mutex_);
			bool was_new = anchor_entry_point_.find(*requests[i].anchor_id) == anchor_entry_point_.end();
			anchor_entry_point_[*requests[i].anchor_id] = *internal_id;
			ARACHNE_LOG_DEBUG(
					"HnswlibIndexGpu::traverseHost: {} anchor_entry_point_[{}] = {} (top-1 result id={}, cache size={})",
					was_new ? "populated" : "updated", *requests[i].anchor_id, *internal_id, top_id, anchor_entry_point_.size());
		}
	}
	return results;
}

std::uint32_t HnswlibIndexGpu::resolveEntryPoint(const TraverseRequest& request) const {
	if (request.anchor_id) {
		std::lock_guard<std::mutex> lock(anchor_cache_mutex_);
		auto it = anchor_entry_point_.find(*request.anchor_id);
		if (it != anchor_entry_point_.end()) {
			ARACHNE_LOG_DEBUG("HnswlibIndexGpu::resolveEntryPoint: cache HIT for anchor_id={} -> entry={}",
												 *request.anchor_id, it->second);
			return it->second;
		}
	}
	std::uint32_t entry = engineGlobalEntryPoint();
	ARACHNE_LOG_DEBUG(
			"HnswlibIndexGpu::resolveEntryPoint: no cached entry point for anchor_id={} -- falling back to global entry "
			"point {}",
			request.anchor_id ? static_cast<long long>(*request.anchor_id) : -1LL, entry);
	return entry;
}

std::vector<TraverseResult> HnswlibIndexGpu::traverseDevice(const std::vector<TraverseRequest>& requests) {
	return TraverseBatchOnDevice(requests);
}

std::vector<TraverseResult> HnswlibIndexGpu::TraverseBatchOnDevice(const std::vector<TraverseRequest>& requests) {
	if (requests.empty()) return {};

	if (metric() == DistanceMetric::Cosine) {
		throw std::logic_error(
				"HnswlibIndexGpu::traverseDevice: DistanceMetric::Cosine is not supported -- hnswlib has no native "
				"Cosine Space (see hnswlib_index.cpp's makeEngine() TODO), same reason the host path rejects it");
	}
	if (controller_ == nullptr) {
		throw std::logic_error("HnswlibIndexGpu::traverseDevice: attachController() was never called");
	}
	for (const TraverseRequest& request : requests) {
		if (request.query.vector.dtype != dtype()) {
			throw std::invalid_argument("HnswlibIndexGpu::traverseDevice: query vector dtype does not match this adapter's");
		}
	}

	// No mutex_ here (see hnswlib_index.hpp's class doc comment): every hnswlib
	// touch point below (engineLevel0Neighbors(), resolveEntryPoint()'s
	// engineGlobalEntryPoint() fallback, engineIsMarkedDeleted(),
	// engineDataPointerFor(), engineHostDistance()) either takes hnswlib's
	// own lock itself or is safe unsynchronized by hnswlib's own design (see
	// hnswlib_index.cpp's TypedHnswEngine). Safety against a concurrently-running Modify is
	// OpScheduler's job (IAdapter::requiresTraverseModifyIsolation()), not
	// this function's -- and since it always admits Traverse concurrently
	// with other Traverse (host or device), this function may now genuinely
	// run in parallel with itself and with traverseHost().

	const std::size_t batch_size = requests.size();
	const std::uint32_t dim = this->dim();
	const std::size_t element_bytes = VectorElementSize(dtype());
	const std::size_t beam_width = std::max<std::size_t>(1, BeamWidth());

	// This worker's own persistent stream/scratch (Controller::workerStream()/
	// workerScratch()) -- neither depends on any particular Region being
	// resident, unlike the old design where the stream came from the entry
	// point's own device lease and an unresolvable entry point meant bailing
	// before round 1 even started. compute_distances_batch() below now
	// resolves residency per candidate (including every request's entry
	// point, via its own first call) instead of this function needing a
	// special case up front for it. See requiredScratchBytesPerWorker()'s
	// own doc comment for how `layout` is sized against max_batch_size_.
	cudaStream_t stream = controller_->workerStream();
	void* scratch = controller_->workerScratch();
	ScratchLayout layout = ComputeScratchLayout(dim, element_bytes, beam_width, M(), max_batch_size_);

	// Per-request search state -- exactly the single-query design's old
	// locals (visited/candidate_set/top_candidates/touched_regions), now one
	// instance per request in `requests` instead of one instance for the
	// single query this function used to handle. `done` marks a request as
	// permanently finished contributing candidates to a round: either its
	// candidate_set emptied out, or its own best-first stop condition
	// tripped (the two ways the old single-query `while` loop used to exit,
	// see the round loop below) -- once true, a request costs nothing in any
	// later round, not even an idle GPU lane, since finished requests are
	// simply omitted when a round's combined candidate list is built.
	struct QueryState {
		std::vector<bool> visited;
		std::priority_queue<Candidate, std::vector<Candidate>, MinFirst> candidate_set;
		std::priority_queue<Candidate> top_candidates;  // max-heap: top() is farthest
		// Regions touched by this request's walk (see TraverseResult::touched's
		// own doc comment, adapter/index_adapter.hpp): every internal id whose
		// distance actually got computed, GPU or host -- not just the final
		// top-k winners, same richer signal the single-query design reported.
		std::unordered_set<RegionId> touched_regions;
		std::size_t ef = 1;
		bool done = false;
	};
	std::vector<QueryState> states(batch_size);
	for (std::size_t q = 0; q < batch_size; ++q) {
		states[q].visited.assign(capacity(), false);
		std::uint32_t top_k = requests[q].query.top_k == 0 ? 1 : requests[q].query.top_k;
		states[q].ef = top_k;
	}

	// Upload every request's query vector into one contiguous device buffer
	// (`batch_size` `dim`-element vectors, back-to-back) -- gathered into a
	// host staging buffer first and copied in one cudaMemcpyAsync, since the
	// requests' own query vectors aren't necessarily contiguous in host
	// memory. Falls back to a one-off cudaMalloc when there's no scratch, or
	// this call's batch is larger than max_batch_size_ reserved for --
	// same "scratch when it fits, one-off allocation otherwise" convention
	// as compute_distances_batch()'s per-round ptrs/out/query_index buffers
	// below.
	bool queries_owned = scratch == nullptr || batch_size > layout.query_capacity;
	void* device_queries = nullptr;
	if (queries_owned) {
		CheckCuda(cudaMalloc(&device_queries, batch_size * dim * element_bytes), "cudaMalloc(device_queries)");
	} else {
		device_queries = layout.Queries(scratch);
	}
	std::vector<std::byte> query_staging(batch_size * dim * element_bytes);
	for (std::size_t q = 0; q < batch_size; ++q) {
		std::memcpy(query_staging.data() + q * dim * element_bytes, requests[q].query.vector.data, dim * element_bytes);
	}
	CheckCuda(
			cudaMemcpyAsync(device_queries, query_staging.data(), query_staging.size(), cudaMemcpyHostToDevice, stream),
			"cudaMemcpyAsync(queries)");

	// Resolves each of `ids[i]`'s Region and computes its distance against
	// its own query (`query_indices[i]` says which of `requests`/
	// device_queries that is): on GPU, every resident id passed to one call
	// is batched into ONE combined kernel launch regardless of which request
	// it belongs to (LaunchDistanceKernel's candidate_query_index argument
	// tells the kernel which query slot each candidate uses -- this is the
	// actual multi-query fusion); on host, one hnswlib fstdistfunc_() call at
	// a time (engineHostDistance()) for the rest. Always returns one
	// distance per id, in the same order as `ids` -- same per-candidate
	// residency fallback the single-query design had (a non-resident
	// candidate never aborts anything, it's just computed on host instead),
	// now shared across every request in the batch instead of computed
	// separately per request.
	auto compute_distances_batch = [&](const std::vector<std::uint32_t>& query_indices,
																			const std::vector<std::uint32_t>& ids) -> std::vector<float> {
		std::vector<float> out(ids.size());
		std::vector<std::size_t> resident_slots;  // index into ids/query_indices/out/resident_device_ptrs
		std::vector<const void*> resident_device_ptrs;
		std::vector<std::uint32_t> resident_query_indices;
		std::vector<gpu::DeviceRegionPool::Lease> leases;
		resident_slots.reserve(ids.size());
		resident_device_ptrs.reserve(ids.size());
		resident_query_indices.reserve(ids.size());
		leases.reserve(ids.size());

		std::size_t host_computed = 0;
		for (std::size_t i = 0; i < ids.size(); ++i) {
			std::uint32_t id = ids[i];
			std::uint32_t q = query_indices[i];
			RegionId region_id = RegionForInternalId(id);
			states[q].touched_regions.insert(region_id);
			const void* vector_host_ptr = engineDataPointerFor(id);  // always available, resident or not

			RegionAccess access = controller_->acquireRegion(region_id);
			if (!access.on_device) {
				out[i] = engineHostDistance(requests[q].query.vector.data, vector_host_ptr);
				++host_computed;
				continue;
			}

			HostRegionView region_host_view = resolveRegion(region_id)->hostView();
			const auto* region_host_base = static_cast<const char*>(region_host_view.ptr);
			std::ptrdiff_t offset = static_cast<const char*>(vector_host_ptr) - region_host_base;
			// The device buffer is NOT a byte-for-byte mirror of the host buffer
			// starting at offset 0 -- Controller::make() prepends a dirty-bitmap
			// header (gpu/dirty_header.hpp) ahead of the Region data whenever
			// subregion_bytes != 0 (which HnswRegion always sets, to one hnswlib
			// record's worth -- see HnswlibIndex::BuildRegions()). Skip past it.
			std::size_t header_bytes = gpu::DirtyHeaderBytes(region_host_view.bytes, region_host_view.subregion_bytes);
			const void* vector_device_ptr =
					static_cast<const char*>(access.device_lease->ptr()) + header_bytes + offset;
			resident_slots.push_back(i);
			resident_device_ptrs.push_back(vector_device_ptr);
			resident_query_indices.push_back(q);
			leases.push_back(std::move(*access.device_lease));
		}

		if (!resident_device_ptrs.empty()) {
			// Use this worker's scratch slice when it's both available and big
			// enough for this round's resident-candidate count; otherwise fall
			// back to a one-off cudaMalloc for just this round (see
			// ScratchLayout/ComputeScratchLayout's own comment for why the
			// scratch case is expected to always be big enough in practice).
			bool use_scratch = scratch != nullptr && resident_device_ptrs.size() <= layout.max_candidates;
			const void** device_ptrs = nullptr;
			float* device_out = nullptr;
			std::uint32_t* device_query_index = nullptr;
			if (use_scratch) {
				device_ptrs = static_cast<const void**>(layout.Ptrs(scratch));
				device_out = static_cast<float*>(layout.Out(scratch));
				device_query_index = static_cast<std::uint32_t*>(layout.QueryIndex(scratch));
			} else {
				if (scratch != nullptr) {
					ARACHNE_LOG_WARN(
							"HnswlibIndexGpu::traverseDevice: round resident-candidate count {} exceeds reserved scratch "
							"capacity {} -- falling back to a one-off cudaMalloc for this round only",
							resident_device_ptrs.size(), layout.max_candidates);
				}
				CheckCuda(cudaMalloc(&device_ptrs, resident_device_ptrs.size() * sizeof(const void*)),
						"cudaMalloc(device_ptrs)");
				CheckCuda(cudaMalloc(&device_out, resident_device_ptrs.size() * sizeof(float)), "cudaMalloc(device_out)");
				CheckCuda(cudaMalloc(&device_query_index, resident_device_ptrs.size() * sizeof(std::uint32_t)),
						"cudaMalloc(device_query_index)");
			}
			CheckCuda(cudaMemcpyAsync(device_ptrs, resident_device_ptrs.data(),
										 resident_device_ptrs.size() * sizeof(const void*), cudaMemcpyHostToDevice, stream),
					"cudaMemcpyAsync(device_ptrs)");
			CheckCuda(cudaMemcpyAsync(device_query_index, resident_query_indices.data(),
										 resident_query_indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream),
					"cudaMemcpyAsync(device_query_index)");

			LaunchDistanceKernel(ToKernelElemType(dtype()), ToKernelDistanceOp(metric()), device_queries, device_query_index,
													 device_ptrs, resident_device_ptrs.size(), dim, device_out, stream);

			std::vector<float> gpu_out(resident_device_ptrs.size());
			CheckCuda(cudaMemcpyAsync(gpu_out.data(), device_out, resident_device_ptrs.size() * sizeof(float),
										 cudaMemcpyDeviceToHost, stream),
					"cudaMemcpyAsync(host_out)");
			CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
			// Safe to free/let leases drop only now that the sync above proved
			// the kernel (and the copies surrounding it) actually finished --
			// only owned (non-scratch) buffers need freeing at all.
			if (!use_scratch) {
				cudaFree(device_ptrs);
				cudaFree(device_out);
				cudaFree(device_query_index);
			}
			for (std::size_t k = 0; k < resident_slots.size(); ++k) out[resident_slots[k]] = gpu_out[k];
		}

		if (host_computed > 0) {
			ARACHNE_LOG_DEBUG("HnswlibIndexGpu::traverseDevice: round -- gpu_computed={} host_computed={}",
												 resident_device_ptrs.size(), host_computed);
		}
		return out;
	};

	// From-scratch re-implementation of hnswlib's bare-bone level-0 greedy
	// search (thirdparty/hnswlib/hnswlib/hnswalg.h's searchBaseLayerST) --
	// see hnswlib_index_gpu.hpp's file overview for why this can't just call
	// hnswlib's own version, and for the hop-synchronized batching this
	// function now runs across every request in `requests` at once instead
	// of one query at a time. `ef` == top_k here (no separate ef parameter
	// threaded through yet -- a known simplification, not a correctness
	// bug: recall may be lower than hnswlib's own ef_ default would give at
	// the same k, not wrong).

	// Round "-1": seed every request's candidate_set with its own entry
	// point -- one combined distance computation across the whole batch,
	// same shape as any later round below.
	std::vector<std::uint32_t> entries(batch_size);
	{
		std::vector<std::uint32_t> seed_query_indices(batch_size);
		for (std::size_t q = 0; q < batch_size; ++q) {
			entries[q] = resolveEntryPoint(requests[q]);
			states[q].visited[entries[q]] = true;
			seed_query_indices[q] = static_cast<std::uint32_t>(q);
		}
		std::vector<float> entry_dist = compute_distances_batch(seed_query_indices, entries);
		for (std::size_t q = 0; q < batch_size; ++q) {
			states[q].candidate_set.push({entry_dist[q], entries[q]});
			if (!engineIsMarkedDeleted(entries[q])) states[q].top_candidates.push({entry_dist[q], entries[q]});
		}
	}

	// Hop-synchronized main loop: every round, each request that isn't
	// `done` yet pops its own frontier (same beam_width/stop-condition rule
	// the single-query design used) and expands it to new candidates: all of
	// this round's new candidates, across every still-active request, are
	// combined into ONE compute_distances_batch() call -- one GPU kernel
	// launch instead of one per request. The loop ends once every request is
	// `done`.
	for (bool any_active = true; any_active;) {
		std::vector<std::uint32_t> round_query_indices;
		std::vector<std::uint32_t> round_ids;

		for (std::size_t q = 0; q < batch_size; ++q) {
			QueryState& state = states[q];
			if (state.done) continue;
			if (state.candidate_set.empty()) {
				state.done = true;
				continue;
			}

			std::vector<Candidate> frontier;
			while (!state.candidate_set.empty() && frontier.size() < beam_width) {
				const Candidate& next = state.candidate_set.top();
				if (state.top_candidates.size() >= state.ef && next.first > state.top_candidates.top().first) break;
				frontier.push_back(next);
				state.candidate_set.pop();
			}
			if (frontier.empty()) {
				state.done = true;  // best-first stop condition tripped -- permanent, see QueryState's own comment
				continue;
			}

			for (const Candidate& current : frontier) {
				for (std::uint32_t neighbor : engineLevel0Neighbors(current.second)) {
					if (state.visited[neighbor]) continue;
					state.visited[neighbor] = true;
					round_query_indices.push_back(static_cast<std::uint32_t>(q));
					round_ids.push_back(neighbor);
				}
			}
			// A request whose frontier produced no *new* candidates this round
			// (every neighbor already visited) isn't `done` -- its
			// candidate_set may still hold more to try next round, same as the
			// single-query design's old `if (to_compute.empty()) continue;`.
		}

		if (!round_ids.empty()) {
			ARACHNE_LOG_DEBUG("HnswlibIndexGpu::traverseDevice: round -- batch_size={} new_candidates={} (beam_width={})",
												 batch_size, round_ids.size(), beam_width);
			std::vector<float> distances = compute_distances_batch(round_query_indices, round_ids);
			for (std::size_t i = 0; i < round_ids.size(); ++i) {
				QueryState& state = states[round_query_indices[i]];
				float d = distances[i];
				std::uint32_t id = round_ids[i];
				if (state.top_candidates.size() < state.ef || d < state.top_candidates.top().first) {
					state.candidate_set.push({d, id});
					if (!engineIsMarkedDeleted(id)) {
						state.top_candidates.push({d, id});
						if (state.top_candidates.size() > state.ef) state.top_candidates.pop();
					}
				}
			}
		}

		any_active = false;
		for (const QueryState& state : states) {
			if (!state.done) {
				any_active = true;
				break;
			}
		}
	}

	if (queries_owned) cudaFree(device_queries);

	std::vector<TraverseResult> results(batch_size);
	for (std::size_t q = 0; q < batch_size; ++q) {
		TraverseResult& result = results[q];
		result.execution_mode = ExecutionMode::GpuOnly;
		// Always true now -- compute_distances_batch() never fails to produce
		// an answer for residency reasons anymore (it falls back to host per
		// candidate instead), so every request's walk always runs to
		// completion. Kept as an explicit field (not left at its default) so a
		// reader isn't left wondering why -- see compute_distances_batch()'s
		// own comment above for what changed and why.
		result.completed_within_scope = true;
		result.touched.regions.assign(states[q].touched_regions.begin(), states[q].touched_regions.end());

		std::vector<Candidate> ordered;
		ordered.reserve(states[q].top_candidates.size());
		while (!states[q].top_candidates.empty()) {
			ordered.push_back(states[q].top_candidates.top());
			states[q].top_candidates.pop();
		}
		std::reverse(ordered.begin(), ordered.end());  // top_candidates drains farthest-first; flip to closer-first
		std::uint32_t top_k = requests[q].query.top_k == 0 ? 1 : requests[q].query.top_k;
		if (ordered.size() > top_k) ordered.resize(top_k);
		result.result.neighbors.reserve(ordered.size());
		for (const auto& [dist, id] : ordered) result.result.neighbors.push_back(Neighbor{engineExternalLabel(id), dist});

		ARACHNE_LOG_DEBUG("HnswlibIndexGpu::traverseDevice: request={} entry={} found={} touched_regions={}", q,
											 entries[q], result.result.neighbors.size(), result.touched.regions.size());
	}
	return results;
}

}  // namespace arachne::index::hnsw
