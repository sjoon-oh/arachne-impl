#include "hnsw_index_anchor_entry.hpp"

#include <utility>

#include "logging.hpp"

namespace arachne::index::hnsw {

HnswIndexAnchorEntry::HnswIndexAnchorEntry(std::uint32_t dim, VectorDType dtype, DistanceMetric metric,
																					 std::size_t capacity, std::size_t vectors_per_region, std::size_t M,
																					 std::size_t ef_construction, std::size_t beam_width)
		: HnswIndexDist(dim, dtype, metric, capacity, vectors_per_region, M, ef_construction),
			beam_width_(beam_width == 0 ? 1 : beam_width) {}

std::vector<TraverseResult> HnswIndexAnchorEntry::traverseHost(const std::vector<TraverseRequest>& requests) {
	std::vector<TraverseResult> results = HnswIndex::traverseHost(requests);
	for (std::size_t i = 0; i < requests.size(); ++i) {
		if (!requests[i].anchor_id || results[i].result.neighbors.empty()) continue;
		VectorId top_id = results[i].result.neighbors.front().id;
		if (std::optional<std::uint32_t> internal_id = engineInternalIdFor(top_id)) {
			std::lock_guard<std::mutex> lock(anchor_cache_mutex_);
			bool was_new = anchor_entry_point_.find(*requests[i].anchor_id) == anchor_entry_point_.end();
			anchor_entry_point_[*requests[i].anchor_id] = *internal_id;
			ARACHNE_LOG_DEBUG(
					"HnswIndexAnchorEntry::traverseHost: {} anchor_entry_point_[{}] = {} (top-1 result id={}, cache "
					"size={})",
					was_new ? "populated" : "updated", *requests[i].anchor_id, *internal_id, top_id,
					anchor_entry_point_.size());
		}
	}
	return results;
}

std::uint32_t HnswIndexAnchorEntry::resolveEntryPoint(const TraverseRequest& request) const {
	if (request.anchor_id) {
		std::lock_guard<std::mutex> lock(anchor_cache_mutex_);
		auto it = anchor_entry_point_.find(*request.anchor_id);
		if (it != anchor_entry_point_.end()) {
			ARACHNE_LOG_DEBUG("HnswIndexAnchorEntry::resolveEntryPoint: cache HIT for anchor_id={} -> entry={}",
												 *request.anchor_id, it->second);
			return it->second;
		}
	}
	std::uint32_t entry = engineGlobalEntryPoint();
	ARACHNE_LOG_DEBUG(
			"HnswIndexAnchorEntry::resolveEntryPoint: no cached entry point for anchor_id={} -- falling back to global "
			"entry point {}",
			request.anchor_id ? static_cast<long long>(*request.anchor_id) : -1LL, entry);
	return entry;
}

}  // namespace arachne::index::hnsw
