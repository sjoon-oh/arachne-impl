// Tests for ASRoutingCacheHnsw, the hnswlib-backed implementation of
// RoutingCache's Anchor-matching contract: ensure() either creates a new
// entry or returns the id of an existing one already within max_distance;
// nearest() finds the closest entry that is within its own stored radius,
// or nullopt; erase() removes an entry, eventually via a background
// compaction once the tombstone ratio crosses max_tombstone_ratio. Coverage
// here is Float32 with both the L2 and Cosine metrics; the full
// (VectorDType, DistanceMetric) matrix for the other supported dtypes lives
// in routing_cache_hnsw_dtype_test.cpp. The last two tests in this file
// exist to exercise realistic scale/dimensionality and concurrent access
// respectively, rather than one specific behavior.

#include "core/routing_cache_hnsw.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <thread>
#include <vector>

namespace {

using arachne::DistanceMetric;
using arachne::ASRoutingCacheHnsw;
using arachne::VectorId;
using arachne::VectorView;

constexpr std::uint32_t kDim = 2;

VectorView View(const std::array<float, kDim>& v) { return VectorView{v.data(), kDim}; }

constexpr float kDefaultMaxDistance = 1e-3f;  // squared L2

TEST(ASRoutingCacheHnswTest, EnsureCreatesAndReturnsNewId) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{1.0f, 2.0f};

	VectorId id = cache.ensure(42, View(v), kDefaultMaxDistance);

	EXPECT_EQ(id, 42u);
}

TEST(ASRoutingCacheHnswTest, EnsureReturnsSameIdForCloseVector) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{1.0f, 2.0f};

	cache.ensure(1, View(v), kDefaultMaxDistance);
	VectorId id = cache.ensure(2, View(v), kDefaultMaxDistance);

	EXPECT_EQ(id, 1u);  // id 2 was never needed; the existing entry was reused
}

TEST(ASRoutingCacheHnswTest, NearestReturnsNulloptWhenEmpty) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> query{0.0f, 0.0f};

	EXPECT_EQ(cache.nearest(View(query)), std::nullopt);
}

TEST(ASRoutingCacheHnswTest, NearestFindsExactMatch) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{5.0f, 5.0f};
	cache.ensure(1, View(v), kDefaultMaxDistance);

	EXPECT_EQ(cache.nearest(View(v)), 1u);
}

TEST(ASRoutingCacheHnswTest, NearestReturnsNulloptBeyondThreshold) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{0.0f, 0.0f};
	cache.ensure(1, View(v), kDefaultMaxDistance);

	std::array<float, kDim> far_query{100.0f, 100.0f};
	EXPECT_EQ(cache.nearest(View(far_query)), std::nullopt);
}

TEST(ASRoutingCacheHnswTest, NearestUsesEachAnchorsOwnRadius) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> tight{0.0f, 0.0f};
	std::array<float, kDim> wide{10.0f, 10.0f};
	cache.ensure(1, View(tight), /*max_distance=*/1e-3f);
	cache.ensure(2, View(wide), /*max_distance=*/50.0f);

	// Closer (raw distance) to the tight anchor, but outside its radius.
	// Never considers the wide anchor even though its radius would cover
	// this query -- only the single hnsw top-1 candidate is checked.
	std::array<float, kDim> query{1.0f, 1.0f};
	EXPECT_EQ(cache.nearest(View(query)), std::nullopt);

	// Within the wide anchor's own radius, and it's the raw-nearest one.
	std::array<float, kDim> near_wide{9.0f, 9.0f};
	EXPECT_EQ(cache.nearest(View(near_wide)), 2u);
}

TEST(ASRoutingCacheHnswTest, CosineMatchesParallelVectorsRegardlessOfMagnitude) {
	ASRoutingCacheHnsw cache(kDim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2,
												 /*M=*/16, /*ef_construction=*/200, DistanceMetric::Cosine);
	std::array<float, kDim> v{1.0f, 0.0f};
	// max_distance in Cosine's units is 1 - dot(unit_x, unit_y); a small
	// positive threshold tolerates near-parallel vectors.
	cache.ensure(1, View(v), /*max_distance=*/1e-3f);

	// Same direction, very different magnitude -- L2 would put these far
	// apart, but cosine distance between them is ~0.
	std::array<float, kDim> same_direction{50.0f, 0.0f};
	EXPECT_EQ(cache.nearest(View(same_direction)), 1u);

	// Orthogonal: cosine distance is 1.0, outside the threshold.
	std::array<float, kDim> orthogonal{0.0f, 1.0f};
	EXPECT_EQ(cache.nearest(View(orthogonal)), std::nullopt);
}

TEST(ASRoutingCacheHnswTest, EraseRemovesEntry) {
	ASRoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{3.0f, 3.0f};
	cache.ensure(1, View(v), kDefaultMaxDistance);

	cache.erase(1);

	EXPECT_EQ(cache.nearest(View(v)), std::nullopt);
}

TEST(ASRoutingCacheHnswTest, EraseOfUnknownIdIsANoop) {
	ASRoutingCacheHnsw cache(kDim);
	EXPECT_NO_THROW(cache.erase(123));
}

TEST(ASRoutingCacheHnswTest, EnsureGrowsBeyondInitialCapacity) {
	ASRoutingCacheHnsw cache(kDim, /*initial_capacity=*/2);

	for (int i = 1; i <= 5; ++i) {
		std::array<float, kDim> v{static_cast<float>(i), static_cast<float>(i)};
		cache.ensure(static_cast<VectorId>(i), View(v), kDefaultMaxDistance);
	}

	for (int i = 1; i <= 5; ++i) {
		std::array<float, kDim> v{static_cast<float>(i), static_cast<float>(i)};
		EXPECT_EQ(cache.nearest(View(v)), static_cast<VectorId>(i)) << "id " << i;
	}
}

TEST(ASRoutingCacheHnswTest, CompactionKeepsSurvivingIdsQueryableAndDropsErasedOnes) {
	// initial_capacity/max_tombstone_ratio chosen so the second erase() below
	// is known to cross the ratio and trigger a compaction.
	ASRoutingCacheHnsw cache(kDim, /*initial_capacity=*/8, /*max_tombstone_ratio=*/0.5);

	std::array<float, kDim> v1{1.0f, 1.0f};
	std::array<float, kDim> v2{2.0f, 2.0f};
	std::array<float, kDim> v3{3.0f, 3.0f};
	std::array<float, kDim> v4{4.0f, 4.0f};

	cache.ensure(1, View(v1), kDefaultMaxDistance);
	cache.ensure(2, View(v2), kDefaultMaxDistance);
	cache.ensure(3, View(v3), kDefaultMaxDistance);
	cache.ensure(4, View(v4), kDefaultMaxDistance);

	cache.erase(2);  // tombstone ratio 1/4, below 0.5 -- no compaction yet
	cache.erase(3);  // tombstone ratio 2/4 == 0.5 -- triggers a background compaction
	cache.waitForCompaction();

	// Survivors must still be reachable by their own vectors after the
	// underlying hnswlib instance got rebuilt.
	EXPECT_EQ(cache.nearest(View(v1)), 1u);
	EXPECT_EQ(cache.nearest(View(v4)), 4u);

	// Erased ids must not resurrect.
	EXPECT_EQ(cache.nearest(View(v2)), std::nullopt);
	EXPECT_EQ(cache.nearest(View(v3)), std::nullopt);
}

TEST(ASRoutingCacheHnswTest, CompactionPreservesEachSurvivorsOwnRadius) {
	ASRoutingCacheHnsw cache(kDim, /*initial_capacity=*/8, /*max_tombstone_ratio=*/0.5);

	std::array<float, kDim> tight{0.0f, 0.0f};
	std::array<float, kDim> wide{10.0f, 10.0f};
	std::array<float, kDim> gone1{20.0f, 20.0f};
	std::array<float, kDim> gone2{30.0f, 30.0f};

	cache.ensure(1, View(tight), /*max_distance=*/1e-3f);
	cache.ensure(2, View(wide), /*max_distance=*/50.0f);
	cache.ensure(3, View(gone1), kDefaultMaxDistance);
	cache.ensure(4, View(gone2), kDefaultMaxDistance);

	cache.erase(3);
	cache.erase(4);  // tombstone ratio 2/4 == 0.5 -- triggers a background compaction
	cache.waitForCompaction();

	// Each survivor's radius (not just its vector) must have migrated with it.
	std::array<float, kDim> query{1.0f, 1.0f};
	EXPECT_EQ(cache.nearest(View(query)), std::nullopt);  // outside anchor 1's tight radius

	std::array<float, kDim> near_wide{9.0f, 9.0f};
	EXPECT_EQ(cache.nearest(View(near_wide)), 2u);  // inside anchor 2's wide radius
}

// Unlike this file's other tiny fixed low-dim vectors, this test builds a
// realistic-scale hnsw graph: a fixed-seed PRNG generates many random float
// vectors at a larger dim, forcing hnswlib resizeIndex growth (like
// EnsureGrowsBeyondInitialCapacity above but at scale), then a chunk is
// erased to also exercise compaction at scale.
TEST(ASRoutingCacheHnswTest, BuildsAndCompactsWithManyRandomHighDimensionalVectors) {
	constexpr std::uint32_t kLargeDim = 100;
	constexpr std::size_t kNumVectors = 300;
	constexpr float kSelfMaxDistance = 1e-3f;  // distance-to-self is always 0

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<std::vector<float>> vectors(kNumVectors);
	for (auto& v : vectors) {
		v.resize(kLargeDim);
		for (float& x : v) x = dist(rng);
	}

	ASRoutingCacheHnsw cache(kLargeDim, /*initial_capacity=*/64, /*max_tombstone_ratio=*/0.3);
	for (std::size_t i = 0; i < kNumVectors; ++i) {
		VectorId id = cache.ensure(static_cast<VectorId>(i) + 1,
																VectorView{vectors[i].data(), kLargeDim}, kSelfMaxDistance);
		ASSERT_EQ(id, static_cast<VectorId>(i) + 1) << "vector " << i;
	}

	for (std::size_t i = 0; i < kNumVectors; ++i) {
		EXPECT_EQ(cache.nearest(VectorView{vectors[i].data(), kLargeDim}), static_cast<VectorId>(i) + 1)
				<< "vector " << i;
	}

	// Erase a third of them -- past max_tombstone_ratio, so this triggers a
	// background compaction of a graph this size.
	for (std::size_t i = 0; i < kNumVectors; i += 3) {
		cache.erase(static_cast<VectorId>(i) + 1);
	}
	cache.waitForCompaction();

	for (std::size_t i = 0; i < kNumVectors; ++i) {
		std::optional<VectorId> found = cache.nearest(VectorView{vectors[i].data(), kLargeDim});
		if (i % 3 == 0) {
			EXPECT_EQ(found, std::nullopt) << "erased vector " << i << " resurrected";
		} else {
			EXPECT_EQ(found, static_cast<VectorId>(i) + 1) << "survivor " << i << " lost after compaction";
		}
	}
}

// Not a correctness oracle by itself -- meant to run under ThreadSanitizer,
// to catch the hazard ASRoutingCacheHnsw's shared_mutex exists to prevent:
// hnswlib's own locking doesn't protect concurrent reads (searchKnn/
// getDataByLabel) against concurrent writes (addPoint/markDelete/resizeIndex).
TEST(ASRoutingCacheHnswTest, ConcurrentEnsureNearestEraseDoesNotRace) {
	constexpr int kThreads = 8;
	constexpr int kOpsPerThread = 200;

	ASRoutingCacheHnsw cache(kDim, /*initial_capacity=*/16, /*max_tombstone_ratio=*/0.3);

	std::vector<std::thread> threads;
	threads.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&cache, t] {
			for (int i = 0; i < kOpsPerThread; ++i) {
				auto id = static_cast<VectorId>(t * kOpsPerThread + i) + 1;
				std::array<float, kDim> v{static_cast<float>(id), static_cast<float>(id)};

				cache.ensure(id, View(v), kDefaultMaxDistance);
				cache.nearest(View(v));
				if (i % 3 == 0) {
					cache.erase(id);
				}
			}
		});
	}
	for (std::thread& th : threads) th.join();
	cache.waitForCompaction();

	SUCCEED();
}

}  // namespace
