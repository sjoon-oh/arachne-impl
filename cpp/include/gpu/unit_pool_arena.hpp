#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <cuda/memory_resource>
#include <cuda_runtime.h>
#include <raft/core/device_resources.hpp>

namespace arachne::gpu {

/// A fixed-size-subregion ("unit") allocator over one large device buffer,
/// preallocated exactly once at construction and never resized. This is
/// what AllocationPolicy::Pooled is actually backed by now (see
/// device_context.hpp) instead of rmm::mr::pool_memory_resource -- a
/// generic pool allocator has no notion of "this live allocation is
/// currently safe to relocate", so it can never do more than best-fit +
/// coalescing on its own. UnitPoolArena additionally exposes the
/// free-extent structure itself (freeExtentsByAddress()) and a raw D2D
/// relocate() primitive, which is exactly the extra surface a
/// CompactionPolicy needs to actually *move* a still-live allocation when
/// best-fit + coalescing alone can't satisfy a request despite there being
/// enough aggregate free space (see compaction_policy.hpp).
///
/// Not internally synchronized -- every method assumes the caller
/// (DeviceRegionPool) already holds whatever lock protects concurrent
/// access, exactly like DeviceRegionPool's own dealings with
/// cuda::mr::any_resource today.
///
/// Layout: the whole buffer is ceil(total_bytes / unit_bytes) contiguous
/// units, each unit_bytes long. Every request is rounded up to a whole
/// number of units (ceil(bytes / unit_bytes)) and always occupies a
/// *contiguous* run of them, so a live allocation's footprint is always
/// expressible as a single {start_unit, unit_count} pair -- cheap enough to
/// relocate as one D2D copy -- at the cost of up to `unit_bytes - 1` bytes
/// of internal fragmentation per allocation:
///
///   base_ptr_
///   |
///   v
///   +--------+--------+--------+--------+--------+--------+--------+
///   | unit 0 | unit 1 | unit 2 | unit 3 | unit 4 | unit 5 | unit 6 |
///   +--------+--------+--------+--------+--------+--------+--------+
///   |<---- allocation A ------>|        |<---- allocation B ------>|
///   |         {0, 3}           |        |         {4, 3}           |
///                               ^
///                        free extent {3, 1}
///
///   free_by_address_: 3 -> 1          (start_unit -> unit_count)
///   free_by_size_:    1 -> 3          (unit_count -> start_unit)
///
/// Free extents are tracked in two indices kept in sync with each other:
/// free_by_address_ (for coalescing a freed/relocated-away range with its
/// address-adjacent neighbors) and free_by_size_ (for O(log n) best-fit
/// lookup and O(1) largestFreeExtent()).
///
/// Pick `unit_bytes` (see DeviceContext's constructor, which owns one
/// UnitPoolArena per MemoryKind) with the granularity trade-off in mind:
/// too large wastes real GPU memory per Region, too small grows the
/// free-extent indices and the number of units a large Region spans.
class UnitPoolArena {
 public:
	struct UnitRange {
		std::uint64_t start_unit = 0;
		std::uint64_t unit_count = 0;

		std::uint64_t end() const { return start_unit + unit_count; }
		bool empty() const { return unit_count == 0; }
	};

	/// Preallocates ceil(total_bytes / unit_bytes) * unit_bytes contiguous
	/// device bytes from `upstream`, once, here -- never again for this
	/// arena's lifetime, and never returned to `upstream` until this arena
	/// is destroyed. Throws std::invalid_argument if unit_bytes == 0.
	UnitPoolArena(cuda::mr::any_resource<cuda::mr::device_accessible>& upstream,
								raft::device_resources& resources, std::size_t unit_bytes, std::size_t total_bytes);
	~UnitPoolArena();

	UnitPoolArena(const UnitPoolArena&) = delete;
	UnitPoolArena& operator=(const UnitPoolArena&) = delete;

	std::size_t unitBytes() const { return unit_bytes_; }
	std::uint64_t totalUnits() const { return total_units_; }
	std::uint64_t totalFreeUnits() const { return total_free_units_; }

	/// Size of the single largest free extent, in units -- 0 if the arena is
	/// completely full. O(1): free_by_size_ is already size-ordered.
	std::uint64_t largestFreeExtent() const;

	void* pointerFor(UnitRange range) const;

	/// Best-fit: the smallest free extent >= required_units is chosen and
	/// exactly required_units claimed from its start (see claim()); any
	/// remainder returns to the free indices. std::nullopt if no *single*
	/// extent is big enough, even if totalFreeUnits() would be -- telling
	/// "genuinely out of space" from "fragmented, fixable by a
	/// CompactionPolicy" apart is DeviceRegionPool's job, not this arena's.
	std::optional<UnitRange> allocateBestFit(std::uint64_t required_units);

	/// Returns `range` to the free set, coalescing with whichever
	/// immediately-adjacent extent(s) are already free. `range` must exactly
	/// match a range this arena handed out (via allocateBestFit() or claim())
	/// and hasn't already reclaimed.
	void free(UnitRange range);

	/// Removes exactly `range` from the free set, splitting the containing
	/// free extent as needed. Throws std::logic_error if `range` isn't
	/// entirely free right now (callers are expected to already know it is,
	/// e.g. from a freeExtentsByAddress() snapshot taken under the same
	/// lock this call runs under). The primitive both allocateBestFit() and
	/// DeviceRegionPool's compaction executor are built on.
	void claim(UnitRange range);

	/// Address-ordered snapshot of every currently-free extent, for
	/// CompactionPolicy::plan() to reason about. A copy, not a live view --
	/// the caller already holds whatever lock keeps this arena stable for
	/// as long as the snapshot matters.
	std::vector<UnitRange> freeExtentsByAddress() const;

	/// Issues a D2D copy of `from.unit_count * unitBytes()` bytes from
	/// `from`'s location to `to`'s, on `stream`. Pure data movement -- does
	/// not touch free/occupied bookkeeping (`to` must already be claim()'d
	/// by the caller, `from` freed only once safe; see DeviceRegionPool's
	/// compaction executor for the exact event-wait ordering used).
	/// `to.unit_count` must be >= `from.unit_count`.
	void relocate(UnitRange from, UnitRange to, cudaStream_t stream) const;

 private:
	void insertFree(std::uint64_t start_unit, std::uint64_t unit_count);
	void eraseFreeExact(std::uint64_t start_unit, std::uint64_t unit_count);

	std::size_t unit_bytes_;
	std::uint64_t total_units_ = 0;
	std::uint64_t total_free_units_ = 0;
	std::byte* base_ptr_ = nullptr;

	cuda::mr::any_resource<cuda::mr::device_accessible>* upstream_;
	raft::device_resources* resources_;

	std::map<std::uint64_t, std::uint64_t> free_by_address_;    // start_unit -> unit_count
	std::multimap<std::uint64_t, std::uint64_t> free_by_size_;  // unit_count -> start_unit
};

}  // namespace arachne::gpu
