#include "core/region_manager.hpp"

#include <stdexcept>

namespace arachne {

void RegionManager::registerRegion(RegionId id, HostRegionView host) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (regions_.find(id) != regions_.end()) return;

	Region region;
	region.id = id;
	region.host = host;
	regions_.emplace(id, region);
}

bool RegionManager::isRegistered(RegionId id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	return regions_.find(id) != regions_.end();
}

Region RegionManager::regionOf(RegionId id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it == regions_.end()) {
		throw std::invalid_argument("RegionManager: region is not registered");
	}
	return it->second;
}

std::vector<RegionId> RegionManager::regionsOf(VectorId anchor_id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = dependencies_.find(anchor_id);
	if (it == dependencies_.end()) return {};
	return std::vector<RegionId>(it->second.begin(), it->second.end());
}

bool RegionManager::addDependency(VectorId anchor_id, RegionId region_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (regions_.find(region_id) == regions_.end()) return false;

	dependents_[region_id].insert(anchor_id);
	dependencies_[anchor_id].insert(region_id);
	return true;
}

bool RegionManager::removeDependency(VectorId anchor_id, RegionId region_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto dep_it = dependencies_.find(anchor_id);
	if (dep_it == dependencies_.end() || dep_it->second.erase(region_id) == 0) return false;
	if (dep_it->second.empty()) dependencies_.erase(dep_it);

	auto dependents_it = dependents_.find(region_id);
	if (dependents_it == dependents_.end()) return true;  // defensive: shouldn't happen

	dependents_it->second.erase(anchor_id);
	if (!dependents_it->second.empty()) return false;

	dependents_.erase(dependents_it);
	return true;
}

std::vector<RegionId> RegionManager::forget(VectorId anchor_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto dep_it = dependencies_.find(anchor_id);
	if (dep_it == dependencies_.end()) return {};

	std::vector<RegionId> orphaned;
	for (RegionId region_id : dep_it->second) {
		auto dependents_it = dependents_.find(region_id);
		if (dependents_it == dependents_.end()) continue;  // defensive: shouldn't happen

		dependents_it->second.erase(anchor_id);
		if (dependents_it->second.empty()) {
			dependents_.erase(dependents_it);
			orphaned.push_back(region_id);
		}
	}
	dependencies_.erase(dep_it);
	return orphaned;
}

void RegionManager::setLease(RegionId id, LeaseHandle lease) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it != regions_.end()) it->second.lease = lease;
}

void RegionManager::setDevice(RegionId id, gpu::DeviceRegionHandle device) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it != regions_.end()) it->second.device = device;
}

void RegionManager::clearResidency(RegionId id) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = regions_.find(id);
	if (it == regions_.end()) return;

	it->second.lease = LeaseHandle{};
	it->second.device = gpu::DeviceRegionHandle{};
}

}  // namespace arachne
