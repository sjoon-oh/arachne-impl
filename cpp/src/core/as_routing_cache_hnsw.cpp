#include "core/routing_cache_hnsw.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <hnswlib/hnswlib.h>

#include "util/distance.hpp"

namespace arachne {

namespace {

/// One concrete (hnswlib Space type, stored element type, hnswlib distance
/// value type) combination -- covers every (VectorDType, DistanceMetric)
/// pair except Cosine (see CosineInstance below; Cosine's normalize step
/// only makes sense for Float32 today). SpaceT is any of hnswlib's
/// L2Space/L2SpaceHalf/L2SpaceI/L2SpaceInt8/InnerProductSpace/
/// InnerProductSpaceHalf/InnerProductSpaceU8/InnerProductSpaceInt8 (all
/// single-arg `SpaceT(size_t dim)` constructors, which is what lets one
/// template cover all eight); ElemT is the raw element type hnswlib stores
/// (float/uint16_t/unsigned char/int8_t); DistT is whatever
/// hnswlib::SpaceInterface<DistT> that Space implements (float for the
/// float/half spaces, int for the uint8/int8 ones) --
/// HierarchicalNSW<DistT> must match it exactly.
template <typename SpaceT, typename ElemT, typename DistT>
class TypedInstance : public ASRoutingCache::RefreshManager {
 public:
	TypedInstance(std::uint32_t dim, std::size_t capacity, std::size_t M, std::size_t ef_construction)
			: space_(dim), index_(&space_, capacity, M, ef_construction, /*random_seed=*/100,
														/*allow_replace_deleted=*/true) {}

	void insert(VectorId id, const void* vector_data, float max_distance) override {
		ensureCapacity();
		index_.addPoint(vector_data, static_cast<hnswlib::labeltype>(id), /*replace_deleted=*/true);
		live_ids_.insert(id);
		max_distance_[id] = max_distance;
	}

	bool erase(VectorId id) override {
		if (live_ids_.erase(id) == 0) return false;

		index_.markDelete(static_cast<hnswlib::labeltype>(id));
		++tombstones_;
		max_distance_.erase(id);
		return true;
	}

	std::optional<VectorId> findNearest(const void* query_data) override {
		if (index_.getCurrentElementCount() == 0) return std::nullopt;

		auto results = index_.searchKnn(query_data, /*k=*/1);
		if (results.empty()) return std::nullopt;

		auto [distance, label] = results.top();
		VectorId id = static_cast<VectorId>(label);
		auto it = max_distance_.find(id);
		if (it == max_distance_.end() || static_cast<float>(distance) > it->second) return std::nullopt;

		return id;
	}

	std::vector<std::byte> rawVectorOf(VectorId id) override {
		return toBytes(index_.template getDataByLabel<ElemT>(static_cast<hnswlib::labeltype>(id)));
	}

	float maxDistanceOf(VectorId id) const override { return max_distance_.at(id); }

	const std::unordered_set<VectorId>& liveIds() const override { return live_ids_; }
	std::size_t liveCount() const override { return live_ids_.size(); }
	std::size_t tombstoneCount() const override { return tombstones_; }

	void forEachLive(const std::function<void(VectorId, const void*, float)>& fn) const override {
		for (VectorId id : live_ids_) {
			std::vector<ElemT> vec = index_.template getDataByLabel<ElemT>(static_cast<hnswlib::labeltype>(id));
			fn(id, vec.data(), max_distance_.at(id));
		}
	}

 private:
	static std::vector<std::byte> toBytes(const std::vector<ElemT>& vec) {
		std::vector<std::byte> bytes(vec.size() * sizeof(ElemT));
		std::memcpy(bytes.data(), vec.data(), bytes.size());
		return bytes;
	}

	void ensureCapacity() {
		if (index_.getCurrentElementCount() < index_.max_elements_) return;
		index_.resizeIndex(index_.max_elements_ * 2);
	}

	SpaceT space_;
	hnswlib::HierarchicalNSW<DistT> index_;
	std::unordered_set<VectorId> live_ids_;
	std::unordered_map<VectorId, float> max_distance_;
	std::size_t tombstones_ = 0;
};

/// Cosine is InnerProduct over L2-normalized vectors, normalized
/// transparently here so callers (and the other Instances above) always
/// pass raw vectors regardless of metric -- there's no dedicated hnswlib
/// Cosine space. Float32 only (see DistanceMetric's doc comment for why):
/// normalizing a quantized int8/uint8 vector and re-quantizing the result
/// loses precision in a way plain float doesn't, and Float16 would need
/// its own half-aware normalize rather than reusing arachne::util::Normalize
/// as-is.
class CosineInstance : public ASRoutingCache::RefreshManager {
 public:
	CosineInstance(std::uint32_t dim, std::size_t capacity, std::size_t M, std::size_t ef_construction)
			: dim_(dim),
				space_(dim),
				index_(&space_, capacity, M, ef_construction, /*random_seed=*/100,
							 /*allow_replace_deleted=*/true) {}

	void insert(VectorId id, const void* vector_data, float max_distance) override {
		ensureCapacity();
		std::vector<float> unit = normalized(vector_data);
		index_.addPoint(unit.data(), static_cast<hnswlib::labeltype>(id), /*replace_deleted=*/true);
		live_ids_.insert(id);
		max_distance_[id] = max_distance;
	}

	bool erase(VectorId id) override {
		if (live_ids_.erase(id) == 0) return false;

		index_.markDelete(static_cast<hnswlib::labeltype>(id));
		++tombstones_;
		max_distance_.erase(id);
		return true;
	}

	std::optional<VectorId> findNearest(const void* query_data) override {
		if (index_.getCurrentElementCount() == 0) return std::nullopt;

		std::vector<float> unit = normalized(query_data);
		auto results = index_.searchKnn(unit.data(), /*k=*/1);
		if (results.empty()) return std::nullopt;

		auto [distance, label] = results.top();
		VectorId id = static_cast<VectorId>(label);
		auto it = max_distance_.find(id);
		if (it == max_distance_.end() || distance > it->second) return std::nullopt;

		return id;
	}

	std::vector<std::byte> rawVectorOf(VectorId id) override {
		std::vector<float> vec = index_.getDataByLabel<float>(static_cast<hnswlib::labeltype>(id));
		std::vector<std::byte> bytes(vec.size() * sizeof(float));
		std::memcpy(bytes.data(), vec.data(), bytes.size());
		return bytes;
	}

	float maxDistanceOf(VectorId id) const override { return max_distance_.at(id); }

	const std::unordered_set<VectorId>& liveIds() const override { return live_ids_; }
	std::size_t liveCount() const override { return live_ids_.size(); }
	std::size_t tombstoneCount() const override { return tombstones_; }

	void forEachLive(const std::function<void(VectorId, const void*, float)>& fn) const override {
		for (VectorId id : live_ids_) {
			std::vector<float> vec = index_.getDataByLabel<float>(static_cast<hnswlib::labeltype>(id));
			fn(id, vec.data(), max_distance_.at(id));
		}
	}

 private:
	std::vector<float> normalized(const void* vector_data) const {
		std::vector<float> unit(dim_);
		util::Normalize(static_cast<const float*>(vector_data), unit.data(), dim_);
		return unit;
	}

	void ensureCapacity() {
		if (index_.getCurrentElementCount() < index_.max_elements_) return;
		index_.resizeIndex(index_.max_elements_ * 2);
	}

	std::uint32_t dim_;
	hnswlib::InnerProductSpace space_;
	hnswlib::HierarchicalNSW<float> index_;
	std::unordered_set<VectorId> live_ids_;
	std::unordered_map<VectorId, float> max_distance_;
	std::size_t tombstones_ = 0;
};

/// Picks the concrete Instance for a (VectorDType, DistanceMetric) pair.
/// The only place that knows the full type matrix; everything else in this
/// file and in routing_cache_hnsw.hpp works through the abstract
/// ASRoutingCache::Instance.
std::unique_ptr<ASRoutingCache::RefreshManager> makeHnswInstance(std::uint32_t dim, std::size_t capacity,
																														 std::size_t M, std::size_t ef_construction,
																														 DistanceMetric metric, VectorDType dtype) {
	if (metric == DistanceMetric::Cosine) {
		if (dtype != VectorDType::Float32) {
			throw std::invalid_argument(
					"ASRoutingCacheHnsw: DistanceMetric::Cosine only supports VectorDType::Float32");
		}
		return std::make_unique<CosineInstance>(dim, capacity, M, ef_construction);
	}

	switch (dtype) {
		case VectorDType::Float32:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedInstance<hnswlib::L2Space, float, float>>(dim, capacity, M,
																																								 ef_construction);
			}
			return std::make_unique<TypedInstance<hnswlib::InnerProductSpace, float, float>>(dim, capacity, M,
																																												 ef_construction);
		case VectorDType::Float16:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedInstance<hnswlib::L2SpaceHalf, std::uint16_t, float>>(
						dim, capacity, M, ef_construction);
			}
			return std::make_unique<TypedInstance<hnswlib::InnerProductSpaceHalf, std::uint16_t, float>>(
					dim, capacity, M, ef_construction);
		case VectorDType::UInt8:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedInstance<hnswlib::L2SpaceI, unsigned char, int>>(dim, capacity, M,
																																												ef_construction);
			}
			return std::make_unique<TypedInstance<hnswlib::InnerProductSpaceU8, unsigned char, int>>(
					dim, capacity, M, ef_construction);
		case VectorDType::Int8:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedInstance<hnswlib::L2SpaceInt8, std::int8_t, int>>(dim, capacity, M,
																																												 ef_construction);
			}
			return std::make_unique<TypedInstance<hnswlib::InnerProductSpaceInt8, std::int8_t, int>>(
					dim, capacity, M, ef_construction);
	}
	throw std::invalid_argument("ASRoutingCacheHnsw: unknown VectorDType");
}

}  // namespace

ASRoutingCacheHnsw::ASRoutingCacheHnsw(std::uint32_t dim, std::size_t initial_capacity,
																		double max_tombstone_ratio, std::size_t M,
																		std::size_t ef_construction, DistanceMetric metric,
																		VectorDType dtype)
		: ASRoutingCache(dim, metric, dtype, initial_capacity, max_tombstone_ratio,
											makeHnswInstance(dim, initial_capacity, M, ef_construction, metric, dtype)),
			M_(M),
			ef_construction_(ef_construction) {}

ASRoutingCacheHnsw::~ASRoutingCacheHnsw() { waitForCompaction(); }

std::unique_ptr<ASRoutingCache::RefreshManager> ASRoutingCacheHnsw::makeRefreshManager(std::size_t capacity) const {
	return makeHnswInstance(dim_, capacity, M_, ef_construction_, metric_, dtype_);
}

}  // namespace arachne
