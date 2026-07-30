#pragma once

#include <cstddef>
#include <cstdint>

namespace arachne::gpu {

/// One dirty-bitmap word is a single 64-bit machine word: small enough that
/// a whole Region's worth of tracking usually stays a handful of words, and
/// exactly the width `atomicOr` on `unsigned long long int` operates on, so
/// a kernel thread marking one subregion dirty never has to coordinate
/// across word boundaries for that single bit -- see LocateDirtyBit().
inline constexpr std::size_t kDirtyWordBytes = sizeof(std::uint64_t);
inline constexpr std::size_t kDirtyBitsPerWord = kDirtyWordBytes * 8;

/// How many kDirtyWordBytes-sized words are needed to carry one dirty bit
/// per `subregion_bytes`-sized chunk of a `region_bytes`-sized Region.
/// Returns 0 if `subregion_bytes` is 0 (see HostRegionView::subregion_bytes
/// -- fine-grained tracking disabled for that Region) or if `region_bytes`
/// is 0 (nothing to track).
constexpr std::size_t DirtyHeaderWords(std::size_t region_bytes, std::size_t subregion_bytes) {
	if (subregion_bytes == 0 || region_bytes == 0) return 0;
	std::size_t subregions = (region_bytes + subregion_bytes - 1) / subregion_bytes;
	return (subregions + kDirtyBitsPerWord - 1) / kDirtyBitsPerWord;
}

/// Byte size of the dirty-bitmap header DirtyHeaderWords() above implies --
/// what Controller::make() will need to add on top of a Region's own
/// `bytes` once GPU allocation for Regions is wired (see its doc comment).
constexpr std::size_t DirtyHeaderBytes(std::size_t region_bytes, std::size_t subregion_bytes) {
	return DirtyHeaderWords(region_bytes, subregion_bytes) * kDirtyWordBytes;
}

/// Where, within a Region's dirty-bitmap header, the bit for the subregion
/// covering `offset_in_region` (a byte offset into the Region's own *data*,
/// not the header) lives: `word_index` selects which kDirtyWordBytes word
/// of the header (word_index * kDirtyWordBytes is that word's own byte
/// offset from the header's start), and `bit_index` (0..63) selects the bit
/// within it -- a kernel marks that subregion dirty with a single
/// `atomicOr` of `(1ull << bit_index)` into that word. `subregion_bytes`
/// must be nonzero (fine-grained tracking must be enabled -- see
/// HostRegionView::subregion_bytes) and `offset_in_region` must be within
/// the Region's own bytes; neither is checked here since this is meant to
/// be cheap enough to inline into a hot kernel-side computation.
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
