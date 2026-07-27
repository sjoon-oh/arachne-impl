#include "core/anchor_manager.hpp"

#include <algorithm>
#include <mutex>

namespace arachne {

std::vector<Stitch> AnchorManager::stitchesOf(VectorId anchor_id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = stitches_.find(anchor_id);
	return it == stitches_.end() ? std::vector<Stitch>{} : it->second;
}

void AnchorManager::addStitch(VectorId anchor_id, RegionId region, LeaseHandle lease,
															 gpu::StitchHandle memory) {
	std::lock_guard<std::mutex> lock(mutex_);
	std::vector<Stitch>& stitches = stitches_[anchor_id];
	bool already_stitched = std::any_of(stitches.begin(), stitches.end(),
																			 [region](const Stitch& s) { return s.region == region; });
	if (!already_stitched) stitches.push_back(Stitch{region, lease, memory});
}

Stitch AnchorManager::removeStitch(VectorId anchor_id, RegionId region) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto map_it = stitches_.find(anchor_id);
	if (map_it == stitches_.end()) return Stitch{};

	std::vector<Stitch>& stitches = map_it->second;
	auto it = std::find_if(stitches.begin(), stitches.end(),
													[region](const Stitch& s) { return s.region == region; });
	if (it == stitches.end()) return Stitch{};

	Stitch removed = *it;
	stitches.erase(it);
	if (stitches.empty()) stitches_.erase(map_it);
	return removed;
}

std::vector<Stitch> AnchorManager::forget(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = stitches_.find(anchor_id);
	if (it == stitches_.end()) return {};

	std::vector<Stitch> removed = std::move(it->second);
	stitches_.erase(it);
	return removed;
}

}  // namespace arachne
