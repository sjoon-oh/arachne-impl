#pragma once

#include <cstddef>
#include <cstdint>

namespace arachne::gpu {

/// A Region's optional dirty-bitmap header: fine-grained tracking of which
/// `subregion_bytes`-sized chunks of a Region's data have been written,
/// packed as one bit per subregion into 64-bit words that precede the
/// Region's own data within its GPU allocation:
///
///   Region's device allocation:
///   +-----------------------+---------------------------------------+
///   | dirty header (words)  | Region data (subregion_bytes chunks)  |
///   +-----------------------+---------------------------------------+
///   <-- DirtyHeaderBytes() -->
///
/// DirtyHeaderBytes() is what Controller::make() adds on top of a Region's
/// own `bytes` to size that allocation. A dirty-bitmap word is a single
/// 64-bit machine word: small enough that a whole Region's tracking usually
/// stays a handful of words, and exactly the width `atomicOr` on
/// `unsigned long long int` operates on, so a kernel thread marking one
/// subregion dirty never has to coordinate across word boundaries for that
/// single bit (see LocateDirtyBit()). Tracking is disabled per-Region by
/// setting subregion_bytes == 0 (see HostRegionView::subregion_bytes) --
/// every function below treats that (and region_bytes == 0) as "0 words,
/// nothing to track".
inline constexpr std::size_t kDirtyWordBytes = sizeof(std::uint64_t);
inline constexpr std::size_t kDirtyBitsPerWord = kDirtyWordBytes * 8;

constexpr std::size_t DirtyHeaderWords(std::size_t region_bytes, std::size_t subregion_bytes) {
	if (subregion_bytes == 0 || region_bytes == 0) return 0;
	std::size_t subregions = (region_bytes + subregion_bytes - 1) / subregion_bytes;
	return (subregions + kDirtyBitsPerWord - 1) / kDirtyBitsPerWord;
}

constexpr std::size_t DirtyHeaderBytes(std::size_t region_bytes, std::size_t subregion_bytes) {
	return DirtyHeaderWords(region_bytes, subregion_bytes) * kDirtyWordBytes;
}

/// Location of the dirty bit for the subregion covering `offset_in_region`
/// (a byte offset into the Region's own *data*, not the header): word_index
/// selects the header word, bit_index (0..63) the bit within it -- a kernel
/// marks it with a single `atomicOr` of `(1ull << bit_index)`. Neither
/// `subregion_bytes != 0` nor `offset_in_region` being in-bounds is checked
/// here, since this is meant to be cheap enough to inline into a hot
/// kernel-side computation.
struct DirtyBitLocation {
	std::size_t word_index;
	unsigned bit_index;
};

constexpr DirtyBitLocation LocateDirtyBit(std::size_t offset_in_region, std::size_t subregion_bytes) {
	std::size_t subregion_index = offset_in_region / subregion_bytes;
	return DirtyBitLocation{subregion_index / kDirtyBitsPerWord,
													 static_cast<unsigned>(subregion_index % kDirtyBitsPerWord)};
}

}  // namespace arachne::gpu
