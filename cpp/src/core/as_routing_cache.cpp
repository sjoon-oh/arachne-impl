#include "core/as_routing_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "logging.hpp"

namespace arachne {

ASRoutingCache::ASRoutingCache(std::uint32_t dim, DistanceMetric metric, VectorDType dtype,
																 std::size_t initial_capacity, double max_tombstone_ratio,
																 std::unique_ptr<RefreshManager> initial_active)
		: RoutingCache(dim, metric, dtype),
			initial_capacity_(initial_capacity),
			max_tombstone_ratio_(max_tombstone_ratio),
			active_(std::move(initial_active)) {}

ASRoutingCache::~ASRoutingCache() {
	if (compaction_thread_.joinable()) compaction_thread_.join();
}

std::optional<VectorId> ASRoutingCache::nearestImpl(const VectorView& query) {
	if (query.dtype != dtype_) {
		throw std::invalid_argument("ASRoutingCache::nearest: VectorView::dtype does not match this cache's dtype");
	}
	std::shared_lock lock(mutex_);
	return active_->findNearest(query.data);
}

VectorId ASRoutingCache::ensure(VectorId id, const VectorView& vector, float max_distance) {
	if (vector.dtype != dtype_) {
		throw std::invalid_argument("ASRoutingCache::ensure: VectorView::dtype does not match this cache's dtype");
	}
	std::unique_lock lock(mutex_);
	if (std::optional<VectorId> existing = active_->findNearest(vector.data)) {
		return *existing;
	}
	active_->insert(id, vector.data, max_distance);
	return id;
}

void ASRoutingCache::erase(VectorId id) {
	bool should_compact = false;
	{
		std::unique_lock lock(mutex_);
		if (!active_->erase(id)) return;

		std::size_t total = active_->liveCount() + active_->tombstoneCount();
		double ratio = total == 0 ? 0.0
															 : static_cast<double>(active_->tombstoneCount()) /
																			 static_cast<double>(total);
		should_compact = ratio >= max_tombstone_ratio_;
	}
	if (should_compact) {
		ARACHNE_LOG_INFO("ASRoutingCache::erase: tombstone ratio reached max_tombstone_ratio_={}, triggering compaction",
											max_tombstone_ratio_);
		triggerCompaction();
	}
}

void ASRoutingCache::waitForCompaction() {
	if (compaction_thread_.joinable()) compaction_thread_.join();
}

void ASRoutingCache::triggerCompaction() {
	bool expected = false;
	if (!compacting_.compare_exchange_strong(expected, true)) {
		ARACHNE_LOG_DEBUG("ASRoutingCache::triggerCompaction: compaction already in flight, skipping");
		return;
	}

	if (compaction_thread_.joinable()) compaction_thread_.join();
	compaction_thread_ = std::thread([this] {
		ARACHNE_LOG_INFO("ASRoutingCache: background compaction started");
		compactImpl();
		compacting_.store(false);
		ARACHNE_LOG_INFO("ASRoutingCache: background compaction finished");
	});
}

void ASRoutingCache::compactImpl() {
	struct Snapshot {
		VectorId id;
		std::vector<std::byte> vector;
		float max_distance;
	};

	const std::size_t vector_bytes = static_cast<std::size_t>(dim_) * VectorElementSize(dtype_);

	// Phase 1: snapshot live ids under a brief shared lock. Concurrent
	// readers can proceed during this phase; a concurrent writer blocks only
	// for this copy, not for the rebuild below.
	std::vector<Snapshot> snapshot;
	{
		std::shared_lock lock(mutex_);
		active_->forEachLive([&](VectorId id, const void* data, float max_distance) {
			const std::byte* bytes = static_cast<const std::byte*>(data);
			snapshot.push_back(Snapshot{id, std::vector<std::byte>(bytes, bytes + vector_bytes), max_distance});
		});
	}

	// Phase 2: build the shadow with no lock held at all -- the expensive
	// part. Both readers and writers of active_ proceed fully concurrently.
	// There is no per-id state left to preserve here (Region dependencies
	// live in Core's RegionManager) -- just (id, vector, max_distance) triples.
	std::size_t shadow_capacity = std::max(snapshot.size() * 2, initial_capacity_);
	std::unique_ptr<RefreshManager> shadow = makeRefreshManager(shadow_capacity);
	std::unordered_set<VectorId> migrated;
	migrated.reserve(snapshot.size());
	for (const Snapshot& entry : snapshot) {
		shadow->insert(entry.id, entry.vector.data(), entry.max_distance);
		migrated.insert(entry.id);
	}

	// Phase 3: reconcile whatever changed on active_ since the snapshot, then
	// swap. Exclusive lock, but bounded by the delta, not by N.
	{
		std::unique_lock lock(mutex_);

		const std::unordered_set<VectorId>& current_ids = active_->liveIds();

		// Erased on active_ while we were building -- drop from the shadow too.
		for (VectorId id : migrated) {
			if (current_ids.find(id) == current_ids.end()) {
				shadow->erase(id);
			}
		}
		// Inserted on active_ while we were building -- carry into the shadow.
		for (VectorId id : current_ids) {
			if (migrated.find(id) == migrated.end()) {
				std::vector<std::byte> bytes = active_->rawVectorOf(id);
				shadow->insert(id, bytes.data(), active_->maxDistanceOf(id));
			}
		}

		ARACHNE_LOG_INFO(
				"ASRoutingCache::compactImpl: rebuilt active index -- {} live id(s) migrated (shadow_capacity={})",
				migrated.size(), shadow_capacity);
		active_ = std::move(shadow);
	}
}

}  // namespace arachne
