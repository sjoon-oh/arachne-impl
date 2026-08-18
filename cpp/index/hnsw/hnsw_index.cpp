#include "hnsw_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <hnswlib/hnswlib.h>

#include "core/controller.hpp"
#include "logging.hpp"

// Implementation of HnswRegion/HnswEngine/HnswIndex -- see hnsw_index.hpp's
// file-level overview for the adapter's role and Region layout.

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
// forward-declares.
// ---------------------------------------------------------------------------

/// Surface HnswIndex (and, via its forwarding helpers, HnswIndexAnchorEntry/
/// HnswIndexDist) drive the concrete engine through. Every method here is a
/// direct, thin call into hnswlib's own public API/public member state --
/// see hnsw_index.hpp's file overview for the "used as-is" guarantee.
class HnswEngine {
 public:
	virtual ~HnswEngine() = default;

	// -- Structural: used by HnswIndex's constructor/loadFrom() to slice Regions.
	virtual void* dataLevel0Memory() const = 0;
	virtual std::size_t sizeDataPerElement() const = 0;
	virtual std::size_t maxElements() const = 0;
	virtual std::size_t maxM() const = 0;
	virtual std::size_t efConstruction() const = 0;

	// -- Level-0 graph accessors (raw, hnswlib-internal-id-addressed) -- what
	// HnswIndexDist's copied search loop (hnsw_index_dist.cpp) needs; also
	// used by HnswIndex::modifyHost()'s own `modified` footprint computation.
	virtual std::uint32_t globalEntryPoint() const = 0;
	virtual const void* dataPointerFor(std::uint32_t internal_id) const = 0;
	virtual std::vector<std::uint32_t> level0Neighbors(std::uint32_t internal_id) const = 0;
	virtual bool isMarkedDeleted(std::uint32_t internal_id) const = 0;
	virtual std::optional<std::uint32_t> internalIdFor(VectorId external_id) const = 0;
	virtual VectorId externalLabel(std::uint32_t internal_id) const = 0;

	// -- Algorithm bridge (label-addressed, i.e. the public-facing shape).
	virtual TraverseResult traverseOne(const TraverseRequest& request) const = 0;
	virtual std::optional<std::uint32_t> insertOne(const ModifyRequest& request) = 0;  // nullopt on failure
	virtual bool deleteOne(const ModifyRequest& request) = 0;                          // false if unknown/already deleted
	virtual void exportTo(const std::string& path) const = 0;
	virtual void loadFrom(const std::string& path, std::size_t max_elements_hint) = 0;
	virtual std::size_t liveCount() const = 0;

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
	std::size_t maxM() const override { return index_.M_; }
	std::size_t efConstruction() const override { return index_.ef_construction_; }
	std::uint32_t globalEntryPoint() const override { return static_cast<std::uint32_t>(index_.enterpoint_node_); }

	const void* dataPointerFor(std::uint32_t internal_id) const override {
		return index_.getDataByInternalId(internal_id);
	}

	std::vector<std::uint32_t> level0Neighbors(std::uint32_t internal_id) const override {
		hnswlib::linklistsizeint* ll = index_.get_linklist0(internal_id);
		std::size_t count = index_.getListCount(ll);
		hnswlib::tableint* neighbors = reinterpret_cast<hnswlib::tableint*>(ll + 1);
		std::vector<std::uint32_t> result;
		result.reserve(count);
		for (std::size_t i = 0; i < count; ++i) result.push_back(static_cast<std::uint32_t>(neighbors[i]));
		return result;
	}

	bool isMarkedDeleted(std::uint32_t internal_id) const override { return index_.isMarkedDeleted(internal_id); }

	VectorId externalLabel(std::uint32_t internal_id) const override {
		return static_cast<VectorId>(index_.getExternalLabel(internal_id));
	}

	std::optional<std::uint32_t> internalIdFor(VectorId external_id) const override {
		auto it = index_.label_lookup_.find(static_cast<hnswlib::labeltype>(external_id));
		if (it == index_.label_lookup_.end()) return std::nullopt;
		return static_cast<std::uint32_t>(it->second);
	}

	TraverseResult traverseOne(const TraverseRequest& request) const override {
		TraverseResult result;
		result.execution_mode = ExecutionMode::Hybrid;
		if (index_.getCurrentElementCount() == 0) {
			result.completed_within_scope = true;  // nothing to find -- trivially "complete"
			return result;
		}
		std::uint32_t top_k = request.query.top_k == 0 ? 1 : request.query.top_k;
		// closer-first: matches Neighbor's expected ascending-distance ordering
		// (see test/stress/stress_index.cpp's ScanOne() for the same convention)
		// without a manual reversal.
		auto closest = index_.searchKnnCloserFirst(request.query.vector.data, top_k);
		result.result.neighbors.reserve(closest.size());
		for (const auto& [dist, label] : closest) {
			result.result.neighbors.push_back(Neighbor{static_cast<VectorId>(label), static_cast<float>(dist)});
		}
		result.completed_within_scope = true;
		return result;
	}

	std::optional<std::uint32_t> insertOne(const ModifyRequest& request) override {
		hnswlib::labeltype label = static_cast<hnswlib::labeltype>(request.record.id);
		try {
			index_.addPoint(request.record.vector.data, label);
		} catch (const std::exception& e) {
			ARACHNE_LOG_WARN("HnswEngine::insertOne: addPoint(label={}) threw: {}", request.record.id, e.what());
			return std::nullopt;
		}
		auto it = index_.label_lookup_.find(label);
		if (it == index_.label_lookup_.end()) return std::nullopt;  // shouldn't happen -- addPoint() just set it
		return static_cast<std::uint32_t>(it->second);
	}

	bool deleteOne(const ModifyRequest& request) override {
		hnswlib::labeltype label = static_cast<hnswlib::labeltype>(request.target);
		try {
			index_.markDelete(label);
		} catch (const std::exception& e) {
			ARACHNE_LOG_WARN("HnswEngine::deleteOne: markDelete(label={}) threw: {}", request.target, e.what());
			return false;
		}
		return true;
	}

	void exportTo(const std::string& path) const override { index_.saveIndex(path); }

	void loadFrom(const std::string& path, std::size_t max_elements_hint) override {
		index_.loadIndex(path, &space_, max_elements_hint);
	}

	std::size_t liveCount() const override { return index_.getCurrentElementCount() - index_.getDeletedCount(); }

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
	// mutable: several of hnswlib's own methods (getCurrentElementCount(),
	// saveIndex(), searchKnnCloserFirst() is const but getCurrentElementCount()
	// isn't, ...) aren't const-qualified even though they're logically
	// read-only -- not something to fix by patching hnswlib (see hnsw_index.hpp's
	// "used as-is" guarantee), so this wrapper absorbs it here instead.
	mutable SpaceT space_;
	mutable hnswlib::HierarchicalNSW<DistT> index_;
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
	BuildRegions();

	ARACHNE_LOG_INFO(
			"HnswIndex: constructed (dim={} dtype={} metric={} capacity={} vectors_per_region={} regions={} M={} "
			"ef_construction={})",
			dim_, static_cast<int>(dtype_), static_cast<int>(metric_), capacity_, vectors_per_region_, regions_.size(),
			M_, ef_construction_);
}

HnswIndex::~HnswIndex() = default;

void HnswIndex::BuildRegions() {
	regions_.clear();
	char* base = static_cast<char*>(engine_->dataLevel0Memory());
	std::size_t record_bytes = engine_->sizeDataPerElement();
	for (std::size_t start = 0; start < capacity_; start += vectors_per_region_) {
		std::size_t count = std::min(vectors_per_region_, capacity_ - start);
		RegionId region_id = static_cast<RegionId>(regions_.size());
		void* ptr = base + start * record_bytes;
		regions_.push_back(std::make_unique<HnswRegion>(region_id, ptr, count * record_bytes, record_bytes));
	}
}

std::uint32_t HnswIndex::resolveEntryPoint(const TraverseRequest& /*request*/) const {
	std::uint32_t entry = engineGlobalEntryPoint();
	ARACHNE_LOG_DEBUG("HnswIndex::resolveEntryPoint: global entry point {} (default strategy)", entry);
	return entry;
}

std::vector<TraverseResult> HnswIndex::traverseHost(const std::vector<TraverseRequest>& requests) {
	std::vector<TraverseResult> results;
	results.reserve(requests.size());
	std::lock_guard<std::mutex> lock(mutex_);
	for (const TraverseRequest& request : requests) {
		TraverseResult result = engine_->traverseOne(request);
		// touched (report.md §7.2 decision (a)): approximated as only the
		// top-k results' own Regions -- hnswlib doesn't expose the full
		// visited-node set through its public API without a source patch,
		// which this class deliberately avoids (see hnsw_index.hpp overview).
		std::unordered_set<RegionId> touched;
		for (const Neighbor& neighbor : result.result.neighbors) {
			if (auto internal_id = engine_->internalIdFor(neighbor.id)) touched.insert(RegionForInternalId(*internal_id));
		}
		result.touched.regions.assign(touched.begin(), touched.end());
		ARACHNE_LOG_DEBUG(
				"HnswIndex::traverseHost: top_k={} found={} touched_regions={} completed_within_scope={}",
				request.query.top_k, result.result.neighbors.size(), result.touched.regions.size(),
				result.completed_within_scope);
		results.push_back(std::move(result));
	}
	return results;
}

std::vector<ModifyResult> HnswIndex::modifyHost(const std::vector<ModifyRequest>& requests) {
	std::vector<ModifyResult> results;
	results.reserve(requests.size());
	std::lock_guard<std::mutex> lock(mutex_);
	for (const ModifyRequest& request : requests) {
		ModifyResult result;
		result.execution_mode = ExecutionMode::Hybrid;
		if (request.op == ModifyOp::Insert) {
			std::optional<std::uint32_t> new_id = engine_->insertOne(request);
			result.ok = new_id.has_value();
			if (new_id) {
				// Conservative over-approximation: the new node's own Region plus
				// its immediate level-0 neighbors' Regions -- see modifyHost()'s
				// doc comment in hnsw_index.hpp for why this may under-count what
				// mutuallyConnectNewElement() actually rewired.
				std::unordered_set<RegionId> regions{RegionForInternalId(*new_id)};
				for (std::uint32_t neighbor : engine_->level0Neighbors(*new_id)) regions.insert(RegionForInternalId(neighbor));
				result.touched.regions.assign(regions.begin(), regions.end());
				result.modified.regions = result.touched.regions;
				// Note (debug aid): these `modified` regions are reported back to
				// Controller in ModifyResult, but nothing downstream currently acts
				// on it for a Region that's already GPU-resident (Controller::dispatch(
				// const ModifyRequest&) has no on_complete hook, unlike the
				// TraverseRequest overload) -- if any of the regions logged here is
				// already promoted, its device copy is now stale and nothing
				// invalidates it. Surfaced at DEBUG for exactly this kind of
				// diagnosis.
				ARACHNE_LOG_DEBUG(
						"HnswIndex::modifyHost: Insert id={} -> internal_id={} modified_regions={} "
						"(caller is responsible for noticing if any of these are GPU-resident)",
						request.record.id, *new_id, result.modified.regions.size());
			} else {
				ARACHNE_LOG_DEBUG("HnswIndex::modifyHost: Insert id={} failed (addPoint() threw, see prior WARN)",
													 request.record.id);
			}
		} else {
			result.ok = engine_->deleteOne(request);
			// markDelete() only flips a bit in the target's own record -- no
			// graph rewiring, so nothing else is touched/modified.
			ARACHNE_LOG_DEBUG("HnswIndex::modifyHost: Delete id={} ok={} (tombstone bit only, region byte-identical "
												 "except that bit -- a GPU-resident copy is stale by exactly one bit until re-synced)",
												 request.target, result.ok);
		}
		results.push_back(std::move(result));
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
	ARACHNE_LOG_INFO("HnswIndex::build: done ({} live)", liveCount());
}

void HnswIndex::registerAllRegions(Controller& controller) {
	for (const auto& region : regions_) controller.registerRegion(region->id(), region->hostView());
	ARACHNE_LOG_INFO("HnswIndex::registerAllRegions: registered {} region(s)", regions_.size());
}

void HnswIndex::exportTo(const std::string& path) const {
	std::lock_guard<std::mutex> lock(mutex_);
	engine_->exportTo(path);
	ARACHNE_LOG_INFO("HnswIndex::exportTo: wrote '{}'", path);
}

void HnswIndex::loadFrom(const std::string& path) {
	std::lock_guard<std::mutex> lock(mutex_);
	// max_elements_hint=0, not capacity_: hnswlib's own loadIndex() only
	// falls back to the *file's* saved max_elements_ when what we pass is
	// smaller than the file's cur_element_count (see thirdparty/hnswlib/
	// hnswlib/hnswalg.h's loadIndex()) -- passing capacity_ directly would
	// silently succeed whenever capacity_ happens to be >= the file's live
	// count, even if it doesn't match what the file was actually saved
	// with, defeating the validation below. Passing 0 instead forces that
	// fallback unconditionally (0 is never >= a nonzero cur_element_count),
	// so engine_->maxElements() below always reflects the file's true own
	// capacity to compare against ours.
	//
	// Caveat: this trick can't distinguish a capacity mismatch on a file
	// exported from an *empty* index (cur_element_count==0) -- 0 is not <
	// 0, so hnswlib keeps our hint (0) instead of falling back, and
	// maxElements() ends up 0 either way regardless of the file's true
	// saved capacity. Narrow edge case, not solved here.
	engine_->loadFrom(path, 0);

	// Validate against this instance's own construction parameters rather
	// than reshape to fit the file -- same convention as
	// StressIndex::loadFrom().
	//
	// Caveat: hnswlib's own binary format never stores `dim` independently
	// (see loadIndex()'s field list) -- it just trusts data_size_ from the
	// *current* space_, so a dim mismatch isn't robustly caught here. Not
	// solved (would need reading the file's own vector bytes and cross
	// checking, or a source patch this directory otherwise avoids).
	if (engine_->maxElements() != capacity_) {
		throw std::invalid_argument("HnswIndex::loadFrom: file's max_elements does not match this adapter's capacity");
	}
	if (engine_->maxM() != M_) {
		throw std::invalid_argument("HnswIndex::loadFrom: file's M does not match this adapter's M");
	}
	if (engine_->efConstruction() != ef_construction_) {
		throw std::invalid_argument("HnswIndex::loadFrom: file's ef_construction does not match this adapter's");
	}

	BuildRegions();  // data_level0_memory_ was realloc'd by loadIndex() -- every old HnswRegion pointer is stale
	// engine_->liveCount() directly (not liveCount()) -- mutex_ is already
	// held by this function, and it's a plain std::mutex, not recursive.
	ARACHNE_LOG_INFO("HnswIndex::loadFrom: loaded '{}' ({} live, {} region(s) rebuilt)", path, engine_->liveCount(),
										regions_.size());
}

std::size_t HnswIndex::liveCount() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return engine_->liveCount();
}

RegionId HnswIndex::RegionForInternalId(std::size_t internal_id) const {
	return static_cast<RegionId>(internal_id / vectors_per_region_);
}

std::uint32_t HnswIndex::engineGlobalEntryPoint() const { return engine_->globalEntryPoint(); }

const void* HnswIndex::engineDataPointerFor(std::uint32_t internal_id) const {
	return engine_->dataPointerFor(internal_id);
}

std::vector<std::uint32_t> HnswIndex::engineLevel0Neighbors(std::uint32_t internal_id) const {
	return engine_->level0Neighbors(internal_id);
}

bool HnswIndex::engineIsMarkedDeleted(std::uint32_t internal_id) const { return engine_->isMarkedDeleted(internal_id); }

VectorId HnswIndex::engineExternalLabel(std::uint32_t internal_id) const { return engine_->externalLabel(internal_id); }

std::optional<std::uint32_t> HnswIndex::engineInternalIdFor(VectorId external_id) const {
	return engine_->internalIdFor(external_id);
}

}  // namespace arachne::index::hnsw
