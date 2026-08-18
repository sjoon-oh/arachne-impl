#include "hnsw_index_dist.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/controller.hpp"
#include "gpu/device_region_pool.hpp"
#include "gpu/dirty_header.hpp"
#include "hnsw_dist_kernel.cuh"
#include "logging.hpp"
#include "types.hpp"

// Implementation of HnswIndexDist::traverseDevice() -- see hnsw_index_dist.hpp
// for the full design overview (why this reimplements the search loop
// instead of reusing hnswlib's, the residency/completed_within_scope
// contract, and the dtype/metric scope -- every (VectorDType, DistanceMetric)
// combination hnswlib itself supports except Cosine, see
// hnsw_dist_kernel.cuh's own overview).

namespace arachne::index::hnsw {

namespace {

void CheckCuda(cudaError_t status, const char* what) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("HnswIndexDist: ") + what + ": " + cudaGetErrorString(status));
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
	throw std::invalid_argument("HnswIndexDist: unknown VectorDType");
}

KernelDistanceOp ToKernelDistanceOp(DistanceMetric metric) {
	return metric == DistanceMetric::L2 ? KernelDistanceOp::L2 : KernelDistanceOp::InnerProduct;
}

}  // namespace

std::vector<TraverseResult> HnswIndexDist::traverseDevice(const std::vector<TraverseRequest>& requests) {
	std::vector<TraverseResult> results;
	results.reserve(requests.size());
	for (const TraverseRequest& request : requests) results.push_back(TraverseOneOnDevice(request));
	return results;
}

TraverseResult HnswIndexDist::TraverseOneOnDevice(const TraverseRequest& request) {
	if (metric() == DistanceMetric::Cosine) {
		throw std::logic_error(
				"HnswIndexDist::traverseDevice: DistanceMetric::Cosine is not supported -- hnswlib has no native "
				"Cosine Space (see hnsw_index.cpp's makeEngine() TODO), same reason the host path rejects it");
	}
	if (request.query.vector.dtype != dtype()) {
		throw std::invalid_argument("HnswIndexDist::traverseDevice: query vector dtype does not match this adapter's");
	}
	if (controller_ == nullptr) {
		throw std::logic_error("HnswIndexDist::traverseDevice: attachController() was never called");
	}

	// Coarse serialization, same convention traverseHost()/modifyHost() use
	// (see HnswIndex's own doc comment on mutex_): simplest-correct choice
	// for this first cut, at the cost of not letting a concurrent
	// traverseHost()/modifyHost() call overlap with this one. Revisit if
	// profiling ever shows this as the bottleneck.
	std::lock_guard<std::mutex> lock(mutex_);

	TraverseResult result;
	result.execution_mode = ExecutionMode::GpuOnly;

	const std::uint32_t dim = this->dim();
	const std::uint32_t top_k = request.query.top_k == 0 ? 1 : request.query.top_k;

	std::uint32_t entry = resolveEntryPoint(request);
	RegionId entry_region_id = RegionForInternalId(entry);
	RegionAccess entry_access = controller_->acquireRegion(entry_region_id);
	if (!entry_access.on_device) {
		// Can't even start within GPU-resident scope -- Controller's own
		// fallback_to_hybrid (routeSearch()) is what retries this as Hybrid.
		ARACHNE_LOG_DEBUG(
				"HnswIndexDist::traverseDevice: bailing before round 1 -- entry={} (region={}) not GPU-resident",
				entry, entry_region_id);
		result.completed_within_scope = false;
		return result;
	}
	cudaStream_t stream = entry_access.device_lease->stream();
	const std::size_t element_bytes = VectorElementSize(dtype());

	void* device_query = nullptr;
	CheckCuda(cudaMalloc(&device_query, dim * element_bytes), "cudaMalloc(device_query)");
	CheckCuda(cudaMemcpyAsync(device_query, request.query.vector.data, dim * element_bytes, cudaMemcpyHostToDevice, stream),
			"cudaMemcpyAsync(query)");

	// Resolves each of `ids`' Region, gathers its vector's device pointer
	// (byte offset computed via host pointer arithmetic against the same
	// Region's host view -- device and host mirror each other byte-for-byte,
	// see adapter/region.hpp's HostRegionView doc comment), and runs one
	// batched kernel launch for the whole group. Returns std::nullopt the
	// moment any one of `ids` turns out not to be GPU-resident (see this
	// function's own completed_within_scope handling above).
	auto compute_distances = [&](const std::vector<std::uint32_t>& ids) -> std::optional<std::vector<float>> {
		std::vector<const void*> host_ptrs;
		host_ptrs.reserve(ids.size());
		std::vector<gpu::DeviceRegionPool::Lease> leases;
		leases.reserve(ids.size());
		for (std::uint32_t id : ids) {
			RegionId region_id = RegionForInternalId(id);
			RegionAccess access = controller_->acquireRegion(region_id);
			if (!access.on_device) {
				ARACHNE_LOG_DEBUG(
						"HnswIndexDist::traverseDevice: mid-walk bail -- candidate id={} (region={}) not GPU-resident, "
						"{} lease(s) already acquired this round will be released",
						id, region_id, leases.size());
				return std::nullopt;
			}
			HostRegionView region_host_view = resolveRegion(region_id)->hostView();
			const auto* region_host_base = static_cast<const char*>(region_host_view.ptr);
			const auto* vector_host_ptr = static_cast<const char*>(engineDataPointerFor(id));
			std::ptrdiff_t offset = vector_host_ptr - region_host_base;
			// The device buffer is NOT a byte-for-byte mirror of the host buffer
			// starting at offset 0 -- Controller::make() prepends a dirty-bitmap
			// header (gpu/dirty_header.hpp) ahead of the Region data whenever
			// subregion_bytes != 0 (which HnswRegion always sets, to one hnswlib
			// record's worth -- see HnswIndex::BuildRegions()). Skip past it.
			std::size_t header_bytes = gpu::DirtyHeaderBytes(region_host_view.bytes, region_host_view.subregion_bytes);
			const void* vector_device_ptr =
					static_cast<const char*>(access.device_lease->ptr()) + header_bytes + offset;
			host_ptrs.push_back(vector_device_ptr);
			leases.push_back(std::move(*access.device_lease));
		}

		const void** device_ptrs = nullptr;
		float* device_out = nullptr;
		CheckCuda(cudaMalloc(&device_ptrs, host_ptrs.size() * sizeof(const void*)), "cudaMalloc(device_ptrs)");
		CheckCuda(cudaMalloc(&device_out, host_ptrs.size() * sizeof(float)), "cudaMalloc(device_out)");
		CheckCuda(cudaMemcpyAsync(device_ptrs, host_ptrs.data(), host_ptrs.size() * sizeof(const void*),
									 cudaMemcpyHostToDevice, stream),
				"cudaMemcpyAsync(device_ptrs)");

		LaunchDistanceKernel(ToKernelElemType(dtype()), ToKernelDistanceOp(metric()), device_query, device_ptrs,
												 host_ptrs.size(), dim, device_out, stream);

		std::vector<float> host_out(host_ptrs.size());
		CheckCuda(
				cudaMemcpyAsync(host_out.data(), device_out, host_ptrs.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
				"cudaMemcpyAsync(host_out)");
		CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
		// Safe to free/let leases drop only now that the sync above proved
		// the kernel (and the copies surrounding it) actually finished.
		cudaFree(device_ptrs);
		cudaFree(device_out);
		return host_out;
	};

	// From-scratch re-implementation of hnswlib's bare-bone level-0 greedy
	// search (thirdparty/hnswlib/hnswlib/hnswalg.h's searchBaseLayerST) --
	// see hnsw_index_dist.hpp's file overview for why this can't just call
	// hnswlib's own version. `ef` == top_k here (no separate ef parameter
	// threaded through yet -- a known simplification, not a correctness
	// bug: recall may be lower than hnswlib's own ef_ default would give at
	// the same k, not wrong).
	std::vector<bool> visited(capacity(), false);
	std::priority_queue<Candidate, std::vector<Candidate>, MinFirst> candidate_set;
	std::priority_queue<Candidate> top_candidates;  // max-heap: top() is farthest
	std::size_t ef = top_k;
	bool scope_ok = true;
	const std::size_t beam_width = std::max<std::size_t>(1, BeamWidth());

	auto entry_dist = compute_distances({entry});
	if (!entry_dist) {
		scope_ok = false;
	} else {
		visited[entry] = true;
		candidate_set.push({(*entry_dist)[0], entry});
		if (!engineIsMarkedDeleted(entry)) top_candidates.push({(*entry_dist)[0], entry});
	}

	while (scope_ok && !candidate_set.empty()) {
		// Pop up to `beam_width` candidates eligible under the same stop
		// condition a single-candidate round would have used (checked against
		// top_candidates as it stood at the *start* of this round, same as the
		// B=1 case) -- widening the beam only changes how many of the eligible
		// candidates are batched into one GPU round-trip together, not which
		// candidates are eligible in the first place. B=1 reduces to exactly
		// the original pop-one-then-stop-check-next-round behavior.
		std::vector<Candidate> frontier;
		while (!candidate_set.empty() && frontier.size() < beam_width) {
			const Candidate& next = candidate_set.top();
			if (top_candidates.size() >= ef && next.first > top_candidates.top().first) break;
			frontier.push_back(next);
			candidate_set.pop();
		}
		if (frontier.empty()) break;

		std::vector<std::uint32_t> to_compute;
		for (const Candidate& current : frontier) {
			for (std::uint32_t neighbor : engineLevel0Neighbors(current.second)) {
				if (visited[neighbor]) continue;
				visited[neighbor] = true;
				to_compute.push_back(neighbor);
			}
		}
		if (to_compute.empty()) continue;

		ARACHNE_LOG_DEBUG("HnswIndexDist::traverseDevice: round -- frontier={} new_candidates={} (beam_width={})",
											 frontier.size(), to_compute.size(), beam_width);
		auto distances = compute_distances(to_compute);
		if (!distances) {
			scope_ok = false;
			break;
		}

		for (std::size_t i = 0; i < to_compute.size(); ++i) {
			float d = (*distances)[i];
			std::uint32_t id = to_compute[i];
			if (top_candidates.size() < ef || d < top_candidates.top().first) {
				candidate_set.push({d, id});
				if (!engineIsMarkedDeleted(id)) {
					top_candidates.push({d, id});
					if (top_candidates.size() > ef) top_candidates.pop();
				}
			}
		}
	}

	cudaFree(device_query);
	result.completed_within_scope = scope_ok;

	std::vector<Candidate> ordered;
	ordered.reserve(top_candidates.size());
	while (!top_candidates.empty()) {
		ordered.push_back(top_candidates.top());
		top_candidates.pop();
	}
	std::reverse(ordered.begin(), ordered.end());  // top_candidates drains farthest-first; flip to closer-first
	if (ordered.size() > top_k) ordered.resize(top_k);
	result.result.neighbors.reserve(ordered.size());
	for (const auto& [dist, id] : ordered) result.result.neighbors.push_back(Neighbor{engineExternalLabel(id), dist});

	ARACHNE_LOG_DEBUG("HnswIndexDist::traverseDevice: entry={} found={} completed_within_scope={}", entry,
										 result.result.neighbors.size(), result.completed_within_scope);
	return result;
}

}  // namespace arachne::index::hnsw
