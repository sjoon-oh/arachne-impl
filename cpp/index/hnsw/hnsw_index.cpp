#include "hnsw_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <hnswlib/hnswlib.h>

#include "core/controller.hpp"
#include "logging.hpp"

// Implementation of HnswRegion/HnswEngine/HnswIndex -- see hnsw_index.hpp's
// file-level overview for the adapter's role, Region layout, and which
// parts of this file are still [SKELETON] stubs vs. structurally real.

namespace arachne::index::hnsw {

// ---------------------------------------------------------------------------
// HnswRegion
// ---------------------------------------------------------------------------

HnswRegion::HnswRegion(RegionId id, void* ptr, std::size_t bytes, std::size_t subregion_bytes)
		: id_(id), host_(HostRegionView{ptr, bytes, subregion_bytes}) {}

// ---------------------------------------------------------------------------
// HnswEngine -- type-erased hnswlib::HierarchicalNSW<DistT> holder. Defined
// here (not in the public header) so hnswlib stays a PRIVATE, impl-only
// dependency of this adapter -- same reasoning as core/routing_cache_hnsw.hpp's
// own forward-declared Instance type. This *is* the class hnsw_index.hpp
// forward-declares; the split into "structural" vs. "algorithm" methods
// below is just documentation grouping, not a base/derived split.
// ---------------------------------------------------------------------------

/// Abstract surface HnswIndex drives the concrete engine through. Split
/// into "structural" accessors (real, needed to slice/register Regions even
/// before traverse/modify are wired up) and "algorithm" methods
/// ([SKELETON]: throw std::logic_error for now, see hnsw_index.hpp's
/// TODO(Phase 1, report.md §6) on HnswIndex::traverseHost()/modifyHost()).
class HnswEngine {
 public:
	virtual ~HnswEngine() = default;

	// -- Structural: real, used by HnswIndex's constructor to slice Regions.
	virtual void* dataLevel0Memory() const = 0;
	virtual std::size_t sizeDataPerElement() const = 0;
	virtual std::size_t maxElements() const = 0;

	// -- Algorithm bridge: [SKELETON] not yet implemented.
	virtual TraverseResult traverseOne(const TraverseRequest& request) const = 0;
	virtual ModifyResult insertOne(const ModifyRequest& request) = 0;
	virtual ModifyResult deleteOne(const ModifyRequest& request) = 0;
	virtual void exportTo(const std::string& path) const = 0;
	virtual void loadFrom(const std::string& path) = 0;

	// -- Bulk build: real (temporary hnswlib-native implementation, see
	// HnswIndex::build()'s own doc comment in hnsw_index.hpp).
	virtual void build(const VectorBatchView& dataset) = 0;
};

namespace {

/// One concrete (hnswlib Space type, stored element type, hnswlib distance
/// value type) combination -- the same SpaceT/ElemT/DistT triple
/// ASRoutingCacheHnsw::TypedInstance (src/core/as_routing_cache_hnsw.cpp)
/// already uses, for the same reason (one template covers all eight
/// non-Cosine hnswlib Space classes).
template <typename SpaceT, typename ElemT, typename DistT>
class TypedHnswEngine final : public HnswEngine {
 public:
	TypedHnswEngine(std::uint32_t dim, std::size_t capacity, std::size_t M, std::size_t ef_construction)
			: space_(dim), index_(&space_, capacity, M, ef_construction) {}

	void* dataLevel0Memory() const override { return index_.data_level0_memory_; }
	std::size_t sizeDataPerElement() const override { return index_.size_data_per_element_; }
	std::size_t maxElements() const override { return index_.max_elements_; }

	TraverseResult traverseOne(const TraverseRequest&) const override {
		throw std::logic_error(
				"HnswIndex::traverseHost: not yet implemented (skeleton, see index/hnsw/report.md §6/§7.2)");
	}

	ModifyResult insertOne(const ModifyRequest&) override {
		throw std::logic_error(
				"HnswIndex::modifyHost (Insert): not yet implemented (skeleton, see index/hnsw/report.md §6/§7.2)");
	}

	ModifyResult deleteOne(const ModifyRequest&) override {
		throw std::logic_error(
				"HnswIndex::modifyHost (Delete): not yet implemented (skeleton, see index/hnsw/report.md §7.1/§7.2)");
	}

	void exportTo(const std::string&) const override {
		throw std::logic_error("HnswIndex::exportTo: not yet implemented (skeleton)");
	}

	void loadFrom(const std::string&) override {
		throw std::logic_error("HnswIndex::loadFrom: not yet implemented (skeleton)");
	}

	// TEMPORARY: goes straight through hnswlib's own public addPoint(), one
	// vector at a time -- see HnswIndex::build()'s doc comment in
	// hnsw_index.hpp for why this is a placeholder rather than the "real"
	// path. `dataset`'s dtype/dim are validated by HnswIndex::build() before
	// this is reached, so ElemT below is already known to match
	// dataset.data's actual encoding.
	void build(const VectorBatchView& dataset) override {
		std::size_t stride = space_.get_data_size();
		const auto* base = static_cast<const std::byte*>(dataset.data);
		for (std::size_t i = 0; i < dataset.count; ++i) {
			hnswlib::labeltype label =
					dataset.ids != nullptr ? static_cast<hnswlib::labeltype>(dataset.ids[i]) : static_cast<hnswlib::labeltype>(i);
			const void* vector_data = base + i * stride;
			index_.addPoint(vector_data, label);
		}
	}

 private:
	SpaceT space_;
	hnswlib::HierarchicalNSW<DistT> index_;
};

/// Picks the concrete engine for a (VectorDType, DistanceMetric) pair --
/// mirrors makeHnswInstance() in src/core/as_routing_cache_hnsw.cpp.
///
/// TODO(Cosine): as_routing_cache_hnsw.cpp's CosineInstance normalizes
/// vectors around a plain InnerProductSpace<float> since hnswlib has no
/// dedicated Cosine space; HnswIndex needs the same wrapper before it can
/// support DistanceMetric::Cosine. Rejected here for now rather than
/// silently mishandled.
std::unique_ptr<HnswEngine> makeEngine(std::uint32_t dim, VectorDType dtype, DistanceMetric metric,
																				std::size_t capacity, std::size_t M, std::size_t ef_construction) {
	if (metric == DistanceMetric::Cosine) {
		throw std::invalid_argument(
				"HnswIndex: DistanceMetric::Cosine not yet supported (skeleton, see makeEngine()'s TODO)");
	}

	switch (dtype) {
		case VectorDType::Float32:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedHnswEngine<hnswlib::L2Space, float, float>>(dim, capacity, M,
																																									 ef_construction);
			}
			return std::make_unique<TypedHnswEngine<hnswlib::InnerProductSpace, float, float>>(dim, capacity, M,
																																												ef_construction);
		case VectorDType::Float16:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedHnswEngine<hnswlib::L2SpaceHalf, std::uint16_t, float>>(
						dim, capacity, M, ef_construction);
			}
			return std::make_unique<TypedHnswEngine<hnswlib::InnerProductSpaceHalf, std::uint16_t, float>>(
					dim, capacity, M, ef_construction);
		case VectorDType::UInt8:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedHnswEngine<hnswlib::L2SpaceI, unsigned char, int>>(dim, capacity, M,
																																													ef_construction);
			}
			return std::make_unique<TypedHnswEngine<hnswlib::InnerProductSpaceU8, unsigned char, int>>(
					dim, capacity, M, ef_construction);
		case VectorDType::Int8:
			if (metric == DistanceMetric::L2) {
				return std::make_unique<TypedHnswEngine<hnswlib::L2SpaceInt8, std::int8_t, int>>(dim, capacity, M,
																																												 ef_construction);
			}
			return std::make_unique<TypedHnswEngine<hnswlib::InnerProductSpaceInt8, std::int8_t, int>>(
					dim, capacity, M, ef_construction);
	}
	throw std::invalid_argument("HnswIndex: unknown VectorDType");
}

}  // namespace

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

HnswIndex::HnswIndex(std::uint32_t dim, VectorDType dtype, DistanceMetric metric, std::size_t capacity,
											std::size_t vectors_per_region, std::size_t M, std::size_t ef_construction)
		: dim_(dim),
			dtype_(dtype),
			metric_(metric),
			capacity_(capacity),
			vectors_per_region_(vectors_per_region),
			M_(M),
			ef_construction_(ef_construction) {
	if (vectors_per_region_ == 0) {
		throw std::invalid_argument("HnswIndex: vectors_per_region must be > 0");
	}

	engine_ = makeEngine(dim_, dtype_, metric_, capacity_, M_, ef_construction_);

	char* base = static_cast<char*>(engine_->dataLevel0Memory());
	std::size_t record_bytes = engine_->sizeDataPerElement();
	for (std::size_t start = 0; start < capacity_; start += vectors_per_region_) {
		std::size_t count = std::min(vectors_per_region_, capacity_ - start);
		RegionId region_id = static_cast<RegionId>(regions_.size());
		void* ptr = base + start * record_bytes;
		regions_.push_back(std::make_unique<HnswRegion>(region_id, ptr, count * record_bytes, record_bytes));
	}

	ARACHNE_LOG_INFO(
			"HnswIndex: constructed (dim={} dtype={} metric={} capacity={} vectors_per_region={} regions={} M={} "
			"ef_construction={}) -- [SKELETON] traverse/modify/export/load not yet implemented",
			dim_, static_cast<int>(dtype_), static_cast<int>(metric_), capacity_, vectors_per_region_, regions_.size(),
			M_, ef_construction_);
}

HnswIndex::~HnswIndex() = default;

std::vector<TraverseResult> HnswIndex::traverseHost(const std::vector<TraverseRequest>& requests) {
	ARACHNE_LOG_WARN("HnswIndex::traverseHost: called with {} request(s) -- not yet implemented (skeleton)",
										requests.size());
	std::vector<TraverseResult> results;
	results.reserve(requests.size());
	for (const TraverseRequest& request : requests) {
		std::lock_guard<std::mutex> lock(mutex_);
		results.push_back(engine_->traverseOne(request));
	}
	return results;
}

std::vector<ModifyResult> HnswIndex::modifyHost(const std::vector<ModifyRequest>& requests) {
	ARACHNE_LOG_WARN("HnswIndex::modifyHost: called with {} request(s) -- not yet implemented (skeleton)",
										requests.size());
	std::vector<ModifyResult> results;
	results.reserve(requests.size());
	for (const ModifyRequest& request : requests) {
		std::lock_guard<std::mutex> lock(mutex_);
		results.push_back(request.op == ModifyOp::Insert ? engine_->insertOne(request) : engine_->deleteOne(request));
	}
	return results;
}

IRegion* HnswIndex::resolveRegion(RegionId id) {
	if (id >= regions_.size()) {
		ARACHNE_LOG_WARN("HnswIndex::resolveRegion: unknown region {}", id);
		return nullptr;
	}
	return regions_[id].get();
}

std::vector<RegionId> HnswIndex::allRegions() const {
	std::vector<RegionId> ids;
	ids.reserve(regions_.size());
	for (const auto& region : regions_) ids.push_back(region->id());
	return ids;
}

void HnswIndex::build(const VectorBatchView& dataset) {
	if (dataset.dtype != dtype_) {
		throw std::invalid_argument("HnswIndex::build: dataset dtype does not match this adapter's dtype");
	}
	if (dataset.dim != dim_) {
		throw std::invalid_argument("HnswIndex::build: dataset dim does not match this adapter's dim");
	}
	if (dataset.count > capacity_) {
		throw std::invalid_argument("HnswIndex::build: dataset count exceeds capacity");
	}

	ARACHNE_LOG_INFO(
			"HnswIndex::build: building from {} vector(s) via hnswlib addPoint() (TEMPORARY -- see build()'s doc "
			"comment)",
			dataset.count);
	{
		std::lock_guard<std::mutex> lock(mutex_);
		engine_->build(dataset);
	}
	ARACHNE_LOG_INFO("HnswIndex::build: done");
}

void HnswIndex::registerAllRegions(Controller& controller) {
	for (const auto& region : regions_) controller.registerRegion(region->id(), region->hostView());
	ARACHNE_LOG_INFO("HnswIndex::registerAllRegions: registered {} region(s)", regions_.size());
}

void HnswIndex::exportTo(const std::string& path) const {
	ARACHNE_LOG_WARN("HnswIndex::exportTo: called for '{}' -- not yet implemented (skeleton)", path);
	engine_->exportTo(path);
}

void HnswIndex::loadFrom(const std::string& path) {
	ARACHNE_LOG_WARN("HnswIndex::loadFrom: called for '{}' -- not yet implemented (skeleton)", path);
	engine_->loadFrom(path);
}

std::size_t HnswIndex::liveCount() const {
	throw std::logic_error("HnswIndex::liveCount: not yet implemented (skeleton)");
}

RegionId HnswIndex::RegionForInternalId(std::size_t internal_id) const {
	return static_cast<RegionId>(internal_id / vectors_per_region_);
}

}  // namespace arachne::index::hnsw
