#include "gpu/dirty_header.hpp"

#include <gtest/gtest.h>

namespace {

using arachne::gpu::DirtyBitLocation;
using arachne::gpu::DirtyHeaderBytes;
using arachne::gpu::DirtyHeaderWords;
using arachne::gpu::kDirtyBitsPerWord;
using arachne::gpu::kDirtyWordBytes;
using arachne::gpu::LocateDirtyBit;

TEST(DirtyHeaderTest, ZeroSubregionBytesDisablesTracking) {
	EXPECT_EQ(DirtyHeaderWords(4096, 0), 0u);
	EXPECT_EQ(DirtyHeaderBytes(4096, 0), 0u);
}

TEST(DirtyHeaderTest, ZeroRegionBytesNeedsNoHeader) {
	EXPECT_EQ(DirtyHeaderWords(0, 64), 0u);
}

TEST(DirtyHeaderTest, ExactlyOneWordWhenSubregionsFitInSixtyFourBits) {
	// 64 subregions of 100 bytes each = 6400 bytes, exactly kDirtyBitsPerWord
	// subregions -- should need exactly one word, not two.
	EXPECT_EQ(DirtyHeaderWords(64 * 100, 100), 1u);
	EXPECT_EQ(DirtyHeaderBytes(64 * 100, 100), kDirtyWordBytes);
}

TEST(DirtyHeaderTest, OneExtraSubregionSpillsIntoASecondWord) {
	// 65 subregions needs 65 bits, which doesn't fit in one 64-bit word.
	EXPECT_EQ(DirtyHeaderWords(65 * 100, 100), 2u);
	EXPECT_EQ(DirtyHeaderBytes(65 * 100, 100), 2 * kDirtyWordBytes);
}

TEST(DirtyHeaderTest, PartialTrailingSubregionRoundsUp) {
	// 100-byte subregions over a 150-byte region: subregion 0 covers
	// [0,100), subregion 1 covers [100,150) (partial) -- still 2 subregions,
	// still fits in one word.
	EXPECT_EQ(DirtyHeaderWords(150, 100), 1u);
}

TEST(DirtyHeaderTest, LargeRegionNeedsProportionallyMoreWords) {
	// 1 MiB region, 4 KiB subregions -> 256 subregions -> 4 words (256 bits).
	constexpr std::size_t kMiB = 1024 * 1024;
	constexpr std::size_t kSubregionBytes = 4096;
	EXPECT_EQ(DirtyHeaderWords(kMiB, kSubregionBytes), 4u);
}

TEST(DirtyHeaderTest, LocateDirtyBitFirstSubregionIsWordZeroBitZero) {
	DirtyBitLocation loc = LocateDirtyBit(/*offset_in_region=*/0, /*subregion_bytes=*/100);
	EXPECT_EQ(loc.word_index, 0u);
	EXPECT_EQ(loc.bit_index, 0u);
}

TEST(DirtyHeaderTest, LocateDirtyBitMidSubregionRoundsDownToThatSubregion) {
	// Offset 250 with 100-byte subregions falls in subregion 2 ([200,300)).
	DirtyBitLocation loc = LocateDirtyBit(/*offset_in_region=*/250, /*subregion_bytes=*/100);
	EXPECT_EQ(loc.word_index, 0u);
	EXPECT_EQ(loc.bit_index, 2u);
}

TEST(DirtyHeaderTest, LocateDirtyBitCrossesIntoSecondWordAtBitSixtyFour) {
	// Subregion index 64 is the first bit of the second word.
	std::size_t offset = 64 * 100;
	DirtyBitLocation loc = LocateDirtyBit(offset, /*subregion_bytes=*/100);
	EXPECT_EQ(loc.word_index, 1u);
	EXPECT_EQ(loc.bit_index, 0u);
}

TEST(DirtyHeaderTest, LocateDirtyBitLastBitOfFirstWord) {
	std::size_t offset = 63 * 100;
	DirtyBitLocation loc = LocateDirtyBit(offset, /*subregion_bytes=*/100);
	EXPECT_EQ(loc.word_index, 0u);
	EXPECT_EQ(loc.bit_index, 63u);
}

TEST(DirtyHeaderTest, KDirtyBitsPerWordMatchesWordByteWidth) {
	EXPECT_EQ(kDirtyBitsPerWord, kDirtyWordBytes * 8);
	EXPECT_EQ(kDirtyWordBytes, 8u);  // one atomicOr-able 64-bit word
}

}  // namespace
