#include "arachne/core/routing_cache_hnsw.hpp"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <utility>

#include <hnswlib/hnswlib.h>

namespace arachne {

/// One hnswlib index plus the set of live ids registered into it. Active
/// and Shadow (transiently, inside compactImpl()) are both just Instances
/// -- there is nothing in this type that knows about locking, threading, or
/// the Active/Shadow story; RoutingCacheHnsw owns all of that.
class RoutingCacheHnsw::Instance {
 public:
	Instance(std::uint32_t dim, std::size_t capacity, std::size_t M, std::size_t ef_construction)
			: space_(dim),
				index_(&space_, capacity, M, ef_construction, /*random_seed=*/100,
							 /*allow_replace_deleted=*/true) {}

	void insert(VectorId id, const VectorView& vector) {
		ensureCapacity();
		index_.addPoint(vector.data, static_cast<hnswlib::labeltype>(id), /*replace_deleted=*/true);
		live_ids_.insert(id);
	}

	bool erase(VectorId id) {
		if (live_ids_.erase(id) == 0) return false;

		index_.markDelete(static_cast<hnswlib::labeltype>(id));
		++tombstones_;
		return true;
	}

	std::optional<VectorId> findNearest(const VectorView& query, float max_distance) {
		if (index_.getCurrentElementCount() == 0) return std::nullopt;

		auto results = index_.searchKnn(query.data, /*k=*/1);
		if (results.empty()) return std::nullopt;

		auto [distance, label] = results.top();
		if (distance > max_distance) return std::nullopt;

		return static_cast<VectorId>(label);
	}

	bool contains(VectorId id) const { return live_ids_.count(id) != 0; }

	std::vector<float> vectorOf(VectorId id) {
		return index_.getDataByLabel<float>(static_cast<hnswlib::labeltype>(id));
	}

	std::vector<VectorId> liveIds() const { return {live_ids_.begin(), live_ids_.end()}; }

	std::size_t liveCount() const { return live_ids_.size(); }
	std::size_t tombstoneCount() const { return tombstones_; }

	/// Visits every surviving id and its vector -- used by compactImpl() to
	/// snapshot live state before migrating it into a fresh Instance.
	template <typename Fn>
	void forEachLive(Fn&& fn) const {
		for (VectorId id : live_ids_) {
			auto vec = index_.getDataByLabel<float>(static_cast<hnswlib::labeltype>(id));
			fn(id, VectorView{vec.data(), static_cast<std::uint32_t>(vec.size())});
		}
	}

 private:
	void ensureCapacity() {
		if (index_.getCurrentElementCount() < index_.max_elements_) return;
		index_.resizeIndex(index_.max_elements_ * 2);
	}

	hnswlib::L2Space space_;
	hnswlib::HierarchicalNSW<float> index_;
	std::unordered_set<VectorId> live_ids_;
	std::size_t tombstones_ = 0;
};

RoutingCacheHnsw::RoutingCacheHnsw(std::uint32_t dim, std::size_t initial_capacity,
																		float max_distance, double max_tombstone_ratio, std::size_t M,
																		std::size_t ef_construction)
		: dim_(dim),
			initial_capacity_(initial_capacity),
			M_(M),
			ef_construction_(ef_construction),
			max_distance_(max_distance),
			max_tombstone_ratio_(max_tombstone_ratio),
			active_(std::make_unique<Instance>(dim, initial_capacity, M, ef_construction)) {}

RoutingCacheHnsw::~RoutingCacheHnsw() {
	if (compaction_thread_.joinable()) compaction_thread_.join();
}

std::optional<VectorId> RoutingCacheHnsw::nearest(const VectorView& query) {
	std::shared_lock lock(mutex_);
	return active_->findNearest(query, max_distance_);
}

VectorId RoutingCacheHnsw::ensure(VectorId id, const VectorView& vector) {
	std::unique_lock lock(mutex_);
	if (std::optional<VectorId> existing = active_->findNearest(vector, max_distance_)) {
		return *existing;
	}
	active_->insert(id, vector);
	return id;
}

void RoutingCacheHnsw::erase(VectorId id) {
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
	if (should_compact) triggerCompaction();
}

void RoutingCacheHnsw::waitForCompaction() {
	if (compaction_thread_.joinable()) compaction_thread_.join();
}

void RoutingCacheHnsw::triggerCompaction() {
	bool expected = false;
	if (!compacting_.compare_exchange_strong(expected, true)) return;

	if (compaction_thread_.joinable()) compaction_thread_.join();
	compaction_thread_ = std::thread([this] {
		compactImpl();
		compacting_.store(false);
	});
}

void RoutingCacheHnsw::compactImpl() {
	struct Snapshot {
		VectorId id;
		std::vector<float> vector;
	};

	// Phase 1: snapshot live ids under a brief shared lock. Concurrent
	// readers can proceed during this phase; a concurrent writer blocks only
	// for this copy, not for the rebuild below.
	std::vector<Snapshot> snapshot;
	{
		std::shared_lock lock(mutex_);
		active_->forEachLive([&](VectorId id, const VectorView& vector) {
			snapshot.push_back(Snapshot{id, std::vector<float>(vector.data, vector.data + vector.dim)});
		});
	}

	// Phase 2: build the shadow with no lock held at all -- the expensive
	// part. Both readers and writers of active_ proceed fully concurrently.
	// There is no per-id state left to preserve here (Stitches moved to
	// Core's AnchorManager) -- just (id, vector) pairs.
	std::size_t shadow_capacity = std::max(snapshot.size() * 2, initial_capacity_);
	auto shadow = std::make_unique<Instance>(dim_, shadow_capacity, M_, ef_construction_);
	std::unordered_set<VectorId> migrated;
	migrated.reserve(snapshot.size());
	for (const Snapshot& entry : snapshot) {
		shadow->insert(entry.id, VectorView{entry.vector.data(),
																				 static_cast<std::uint32_t>(entry.vector.size())});
		migrated.insert(entry.id);
	}

	// Phase 3: reconcile whatever changed on active_ since the snapshot, then
	// swap. Exclusive lock, but bounded by the delta, not by N.
	{
		std::unique_lock lock(mutex_);

		std::vector<VectorId> current_ids = active_->liveIds();
		std::unordered_set<VectorId> current_id_set(current_ids.begin(), current_ids.end());

		// Erased on active_ while we were building -- drop from the shadow too.
		for (VectorId id : migrated) {
			if (current_id_set.find(id) == current_id_set.end()) {
				shadow->erase(id);
			}
		}
		// Inserted on active_ while we were building -- carry into the shadow.
		for (VectorId id : current_ids) {
			if (migrated.find(id) == migrated.end()) {
				auto vec = active_->vectorOf(id);
				shadow->insert(id, VectorView{vec.data(), static_cast<std::uint32_t>(vec.size())});
			}
		}

		active_ = std::move(shadow);
	}
}

}  // namespace arachne
