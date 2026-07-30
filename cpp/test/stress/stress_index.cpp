#include "stress_index.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

#include "core/controller.hpp"

namespace arachne::stress {

namespace {

// Decodes one dtype-encoded element as a float, purely for distance math --
// the storage format itself is untouched (StressIndex never re-encodes a
// vector; it just memcpy's whatever bytes the caller already provided in
// `dtype`'s format). Deliberately a local, minimal implementation rather
// than reaching into hnswlib's half_utils.h: hnswlib is meant to stay a
// .cpp-only dependency of arachne_core itself, not something test code
// reaches into directly (see routing_cache_hnsw_dtype_test.cpp's own local
// ToHalfBits() for the same reasoning, mirrored here for the decode
// direction). Subnormal half inputs are flushed to zero -- adequate for
// distance-ordering purposes, not a general-purpose codec.
float ElementToFloat(const void* base, std::uint32_t index, VectorDType dtype) {
	switch (dtype) {
		case VectorDType::Int8:
			return static_cast<float>(static_cast<const std::int8_t*>(base)[index]);
		case VectorDType::UInt8:
			return static_cast<float>(static_cast<const std::uint8_t*>(base)[index]);
		case VectorDType::Float16: {
			std::uint16_t h = static_cast<const std::uint16_t*>(base)[index];
			std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
			std::uint32_t exp = (h >> 10) & 0x1Fu;
			std::uint32_t mant = h & 0x3FFu;
			std::uint32_t bits;
			if (exp == 0) {
				bits = sign;  // zero or subnormal -- flushed to zero
			} else if (exp == 0x1Fu) {
				bits = sign | 0x7F800000u | (mant << 13);  // inf/NaN
			} else {
				bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
			}
			float f;
			std::memcpy(&f, &bits, sizeof(f));
			return f;
		}
		case VectorDType::Float32:
			return static_cast<const float*>(base)[index];
	}
	return 0.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// StressRegion
// ---------------------------------------------------------------------------

StressRegion::StressRegion(RegionId id, void* ptr, std::size_t bytes, std::size_t subregion_bytes) : id_(id) {
	host_.ptr = ptr;
	host_.bytes = bytes;
	host_.subregion_bytes = subregion_bytes;
}

// ---------------------------------------------------------------------------
// StressIndex
// ---------------------------------------------------------------------------

StressIndex::StressIndex(std::uint32_t dim, VectorDType dtype, std::size_t capacity, std::size_t vectors_per_region)
		: dim_(dim),
			dtype_(dtype),
			element_size_(VectorElementSize(dtype)),
			vectors_per_region_(vectors_per_region),
			capacity_(capacity),
			buffer_(capacity * dim * element_size_),
			deleted_(capacity, false) {
	if (vectors_per_region_ == 0) throw std::invalid_argument("StressIndex: vectors_per_region must be nonzero");

	std::size_t num_regions = (capacity_ + vectors_per_region_ - 1) / vectors_per_region_;
	regions_.reserve(num_regions);
	for (std::size_t i = 0; i < num_regions; ++i) {
		std::size_t region_vectors = std::min(vectors_per_region_, capacity_ - i * vectors_per_region_);
		void* ptr = buffer_.data() + i * vectors_per_region_ * dim_ * element_size_;
		std::size_t bytes = region_vectors * dim_ * element_size_;
		// RegionId 0 is reserved (see e.g. LeaseHandle::valid()'s epoch != 0
		// convention elsewhere in Core) -- ids here start at 1.
		regions_.push_back(
				std::make_unique<StressRegion>(static_cast<RegionId>(i + 1), ptr, bytes, /*subregion_bytes=*/dim_ * element_size_));
	}
}

StressIndex::~StressIndex() = default;

const void* StressIndex::SlotPtr(std::size_t slot) const { return buffer_.data() + slot * dim_ * element_size_; }
void* StressIndex::SlotPtr(std::size_t slot) { return buffer_.data() + slot * dim_ * element_size_; }

RegionId StressIndex::RegionForSlot(std::size_t slot) const { return regions_[slot / vectors_per_region_]->id(); }

float StressIndex::DistanceSquared(const void* query, std::size_t slot) const {
	const void* stored = SlotPtr(slot);
	float sum = 0.0f;
	for (std::uint32_t d = 0; d < dim_; ++d) {
		float diff = ElementToFloat(query, d, dtype_) - ElementToFloat(stored, d, dtype_);
		sum += diff * diff;
	}
	return sum;
}

TraverseResult StressIndex::ScanOne(const TraverseRequest& request) const {
	std::vector<Candidate> candidates;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		candidates.reserve(id_to_slot_.size());
		for (const auto& [id, slot] : id_to_slot_) {
			if (deleted_[slot]) continue;
			candidates.push_back(Candidate{DistanceSquared(request.query.vector.data, slot), id, slot});
		}
	}

	std::uint32_t top_k = request.query.top_k == 0 ? 1 : request.query.top_k;
	std::size_t n = std::min<std::size_t>(top_k, candidates.size());
	std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(n), candidates.end(),
										 [](const Candidate& a, const Candidate& b) { return a.dist2 < b.dist2; });

	TraverseResult result;
	std::set<RegionId> touched;
	for (std::size_t i = 0; i < n; ++i) {
		result.result.neighbors.push_back(Neighbor{candidates[i].id, candidates[i].dist2});
		touched.insert(RegionForSlot(candidates[i].slot));
	}
	result.touched.regions.assign(touched.begin(), touched.end());
	result.completed_within_scope = true;
	return result;
}

ModifyResult StressIndex::InsertOne(const ModifyRequest& request) {
	ModifyResult result;
	std::size_t slot;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (next_free_slot_ >= capacity_) {
			result.ok = false;  // out of capacity -- caller sized capacity_ too small for this run
			return result;
		}
		slot = next_free_slot_++;
		std::memcpy(SlotPtr(slot), request.record.vector.data, dim_ * element_size_);
		id_to_slot_[request.record.id] = slot;
		deleted_[slot] = false;
	}

	RegionId region = RegionForSlot(slot);
	result.ok = true;
	result.touched.regions = {region};
	result.modified.regions = {region};
	return result;
}

ModifyResult StressIndex::DeleteOne(const ModifyRequest& request) {
	ModifyResult result;
	std::size_t slot;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = id_to_slot_.find(request.target);
		if (it == id_to_slot_.end()) {
			result.ok = false;
			return result;
		}
		slot = it->second;
		deleted_[slot] = true;
		id_to_slot_.erase(it);
	}

	RegionId region = RegionForSlot(slot);
	result.ok = true;
	result.touched.regions = {region};
	result.modified.regions = {region};
	return result;
}

std::vector<TraverseResult> StressIndex::traverseHost(const std::vector<TraverseRequest>& requests) {
	std::vector<TraverseResult> results;
	results.reserve(requests.size());
	for (const TraverseRequest& request : requests) results.push_back(ScanOne(request));
	return results;
}

std::vector<TraverseResult> StressIndex::traverseDevice(const std::vector<TraverseRequest>& requests) {
	return traverseHost(requests);
}

std::vector<ModifyResult> StressIndex::modifyDevice(const std::vector<ModifyRequest>& requests) {
	return modifyHost(requests);
}

std::vector<ModifyResult> StressIndex::modifyHost(const std::vector<ModifyRequest>& requests) {
	std::vector<ModifyResult> results;
	results.reserve(requests.size());
	for (const ModifyRequest& request : requests) {
		results.push_back(request.op == ModifyOp::Insert ? InsertOne(request) : DeleteOne(request));
	}
	return results;
}

IRegion* StressIndex::resolveRegion(RegionId id) {
	if (id == 0 || id > regions_.size()) return nullptr;
	return regions_[id - 1].get();
}

std::vector<RegionId> StressIndex::allRegions() const {
	std::vector<RegionId> ids;
	ids.reserve(regions_.size());
	for (const auto& region : regions_) ids.push_back(region->id());
	return ids;
}

void StressIndex::registerAllRegions(Controller& controller) {
	for (const auto& region : regions_) controller.registerRegion(region->id(), region->hostView());
}

std::size_t StressIndex::liveCount() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return id_to_slot_.size();
}

// ---------------------------------------------------------------------------

std::vector<Neighbor> BruteForceGroundTruth(const StressIndex& index, const VectorView& query, std::uint32_t top_k) {
	struct Candidate {
		float dist2;
		VectorId id;
	};
	std::vector<Candidate> candidates;
	{
		std::lock_guard<std::mutex> lock(index.mutex_);
		candidates.reserve(index.id_to_slot_.size());
		for (const auto& [id, slot] : index.id_to_slot_) {
			if (index.deleted_[slot]) continue;
			float sum = 0.0f;
			const void* stored = index.SlotPtr(slot);
			for (std::uint32_t d = 0; d < index.dim_; ++d) {
				float diff = ElementToFloat(query.data, d, index.dtype_) - ElementToFloat(stored, d, index.dtype_);
				sum += diff * diff;
			}
			candidates.push_back(Candidate{sum, id});
		}
	}
	std::size_t n = std::min<std::size_t>(top_k == 0 ? 1 : top_k, candidates.size());
	std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(n), candidates.end(),
										 [](const Candidate& a, const Candidate& b) { return a.dist2 < b.dist2; });
	std::vector<Neighbor> result;
	result.reserve(n);
	for (std::size_t i = 0; i < n; ++i) result.push_back(Neighbor{candidates[i].id, candidates[i].dist2});
	return result;
}

}  // namespace arachne::stress
