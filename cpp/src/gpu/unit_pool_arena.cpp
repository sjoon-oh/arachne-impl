#include "gpu/unit_pool_arena.hpp"

#include <rmm/aligned.hpp>

#include <stdexcept>
#include <utility>

namespace arachne::gpu {

namespace {
std::uint64_t CeilDivUnits(std::size_t bytes, std::size_t unit_bytes) {
	return (static_cast<std::uint64_t>(bytes) + unit_bytes - 1) / unit_bytes;
}
}  // namespace

UnitPoolArena::UnitPoolArena(cuda::mr::any_resource<cuda::mr::device_accessible>& upstream,
															raft::device_resources& resources, std::size_t unit_bytes, std::size_t total_bytes)
		: unit_bytes_(unit_bytes), upstream_(&upstream), resources_(&resources) {
	if (unit_bytes == 0) {
		throw std::invalid_argument("UnitPoolArena: unit_bytes must be > 0");
	}
	total_units_ = CeilDivUnits(total_bytes, unit_bytes);
	if (total_units_ > 0) {
		std::size_t rounded_bytes = total_units_ * unit_bytes_;
		base_ptr_ = static_cast<std::byte*>(
				upstream_->allocate(resources_->get_stream(), rounded_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT));
		insertFree(0, total_units_);
	}
}

UnitPoolArena::~UnitPoolArena() {
	if (base_ptr_ != nullptr) {
		upstream_->deallocate(resources_->get_stream(), base_ptr_, total_units_ * unit_bytes_,
													 rmm::CUDA_ALLOCATION_ALIGNMENT);
	}
}

std::uint64_t UnitPoolArena::largestFreeExtent() const {
	if (free_by_size_.empty()) return 0;
	return free_by_size_.rbegin()->first;
}

void* UnitPoolArena::pointerFor(UnitRange range) const {
	return base_ptr_ + range.start_unit * unit_bytes_;
}

void UnitPoolArena::insertFree(std::uint64_t start_unit, std::uint64_t unit_count) {
	if (unit_count == 0) return;
	free_by_address_.emplace(start_unit, unit_count);
	free_by_size_.emplace(unit_count, start_unit);
	total_free_units_ += unit_count;
}

void UnitPoolArena::eraseFreeExact(std::uint64_t start_unit, std::uint64_t unit_count) {
	free_by_address_.erase(start_unit);
	auto [begin, end] = free_by_size_.equal_range(unit_count);
	for (auto it = begin; it != end; ++it) {
		if (it->second == start_unit) {
			free_by_size_.erase(it);
			break;
		}
	}
	total_free_units_ -= unit_count;
}

// Removes `range` from whichever single free extent contains it, splitting
// that extent into a leftover prefix and/or suffix as needed:
//
//   [------------- free extent -------------]
//   [ prefix ][****** range (claimed) ******][ suffix ]
//              (both reinserted as free, if non-empty)
//
// The containing extent is located, fully erased, then 0-2 pieces are
// reinserted -- never a partial/in-place edit -- so free_by_address_ and
// free_by_size_ (which must always agree) stay trivially in sync.
void UnitPoolArena::claim(UnitRange range) {
	if (range.empty()) return;

	// The containing free extent is the last entry whose start_unit <= range.start_unit.
	auto it = free_by_address_.upper_bound(range.start_unit);
	if (it == free_by_address_.begin()) {
		throw std::logic_error("UnitPoolArena::claim: range is not free");
	}
	--it;
	std::uint64_t extent_start = it->first;
	std::uint64_t extent_count = it->second;
	if (range.start_unit < extent_start || range.end() > extent_start + extent_count) {
		throw std::logic_error("UnitPoolArena::claim: range is not entirely free");
	}

	eraseFreeExact(extent_start, extent_count);

	std::uint64_t prefix_count = range.start_unit - extent_start;
	if (prefix_count > 0) insertFree(extent_start, prefix_count);

	std::uint64_t suffix_start = range.end();
	std::uint64_t suffix_count = (extent_start + extent_count) - suffix_start;
	if (suffix_count > 0) insertFree(suffix_start, suffix_count);
}

std::optional<UnitPoolArena::UnitRange> UnitPoolArena::allocateBestFit(std::uint64_t required_units) {
	if (required_units == 0) return UnitRange{0, 0};

	auto it = free_by_size_.lower_bound(required_units);
	if (it == free_by_size_.end()) return std::nullopt;

	UnitRange range{it->second, required_units};
	claim(range);
	return range;
}

// Returns `range` to the free set, coalescing with an immediately-adjacent
// free extent on either side (at most one on each, since free extents can
// never themselves be adjacent -- they'd already have been merged):
//
//   [ left neighbor? ][****** range ******][ right neighbor? ]
//   `-------------------- merged, single insertFree() --------------------'
//
// Absorbing a neighbor means erasing its old (start,count) entry before the
// merged range is (re)inserted -- never leaving a stale entry behind.
void UnitPoolArena::free(UnitRange range) {
	if (range.empty()) return;

	std::uint64_t start_unit = range.start_unit;
	std::uint64_t unit_count = range.unit_count;

	// Left neighbor: a free extent that ends exactly where `range` begins.
	auto left_it = free_by_address_.upper_bound(start_unit);
	if (left_it != free_by_address_.begin()) {
		auto prev = std::prev(left_it);
		if (prev->first + prev->second == start_unit) {
			start_unit = prev->first;
			unit_count += prev->second;
			eraseFreeExact(prev->first, prev->second);
		}
	}

	// Right neighbor: a free extent that begins exactly where `range` (or the
	// already-left-merged extent) ends.
	auto right_it = free_by_address_.find(start_unit + unit_count);
	if (right_it != free_by_address_.end()) {
		unit_count += right_it->second;
		eraseFreeExact(right_it->first, right_it->second);
	}

	insertFree(start_unit, unit_count);
}

std::vector<UnitPoolArena::UnitRange> UnitPoolArena::freeExtentsByAddress() const {
	std::vector<UnitRange> extents;
	extents.reserve(free_by_address_.size());
	for (const auto& [start_unit, unit_count] : free_by_address_) {
		extents.push_back(UnitRange{start_unit, unit_count});
	}
	return extents;
}

void UnitPoolArena::relocate(UnitRange from, UnitRange to, cudaStream_t stream) const {
	if (to.unit_count < from.unit_count) {
		throw std::invalid_argument("UnitPoolArena::relocate: destination range is smaller than source");
	}
	cudaMemcpyAsync(pointerFor(to), pointerFor(from), from.unit_count * unit_bytes_, cudaMemcpyDeviceToDevice,
									 stream);
}

}  // namespace arachne::gpu
