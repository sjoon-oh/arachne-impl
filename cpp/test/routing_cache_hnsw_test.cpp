#include "arachne/core/routing_cache_hnsw.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

namespace {

using arachne::RoutingCacheHnsw;
using arachne::VectorId;
using arachne::VectorView;

constexpr std::uint32_t kDim = 2;

VectorView View(const std::array<float, kDim>& v) { return VectorView{v.data(), kDim}; }

TEST(RoutingCacheHnswTest, EnsureCreatesAndReturnsNewId) {
	RoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{1.0f, 2.0f};

	VectorId id = cache.ensure(42, View(v));

	EXPECT_EQ(id, 42u);
}

TEST(RoutingCacheHnswTest, EnsureReturnsSameIdForCloseVector) {
	RoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{1.0f, 2.0f};

	cache.ensure(1, View(v));
	VectorId id = cache.ensure(2, View(v));

	EXPECT_EQ(id, 1u);  // id 2 was never needed; the existing entry was reused
}

TEST(RoutingCacheHnswTest, NearestReturnsNulloptWhenEmpty) {
	RoutingCacheHnsw cache(kDim);
	std::array<float, kDim> query{0.0f, 0.0f};

	EXPECT_EQ(cache.nearest(View(query)), std::nullopt);
}

TEST(RoutingCacheHnswTest, NearestFindsExactMatch) {
	RoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{5.0f, 5.0f};
	cache.ensure(1, View(v));

	EXPECT_EQ(cache.nearest(View(v)), 1u);
}

TEST(RoutingCacheHnswTest, NearestReturnsNulloptBeyondThreshold) {
	RoutingCacheHnsw cache(kDim);  // default max_distance = 1e-3 (squared L2)
	std::array<float, kDim> v{0.0f, 0.0f};
	cache.ensure(1, View(v));

	std::array<float, kDim> far_query{100.0f, 100.0f};
	EXPECT_EQ(cache.nearest(View(far_query)), std::nullopt);
}

TEST(RoutingCacheHnswTest, EraseRemovesEntry) {
	RoutingCacheHnsw cache(kDim);
	std::array<float, kDim> v{3.0f, 3.0f};
	cache.ensure(1, View(v));

	cache.erase(1);

	EXPECT_EQ(cache.nearest(View(v)), std::nullopt);
}

TEST(RoutingCacheHnswTest, EraseOfUnknownIdIsANoop) {
	RoutingCacheHnsw cache(kDim);
	EXPECT_NO_THROW(cache.erase(123));
}

TEST(RoutingCacheHnswTest, EnsureGrowsBeyondInitialCapacity) {
	RoutingCacheHnsw cache(kDim, /*initial_capacity=*/2);

	for (int i = 1; i <= 5; ++i) {
		std::array<float, kDim> v{static_cast<float>(i), static_cast<float>(i)};
		cache.ensure(static_cast<VectorId>(i), View(v));
	}

	for (int i = 1; i <= 5; ++i) {
		std::array<float, kDim> v{static_cast<float>(i), static_cast<float>(i)};
		EXPECT_EQ(cache.nearest(View(v)), static_cast<VectorId>(i)) << "id " << i;
	}
}

TEST(RoutingCacheHnswTest, CompactionKeepsSurvivingIdsQueryableAndDropsErasedOnes) {
	// initial_capacity/max_tombstone_ratio chosen so the second erase() below
	// is known to cross the ratio and trigger a compaction.
	RoutingCacheHnsw cache(kDim, /*initial_capacity=*/8, /*max_distance=*/1e-3f,
												 /*max_tombstone_ratio=*/0.5);

	std::array<float, kDim> v1{1.0f, 1.0f};
	std::array<float, kDim> v2{2.0f, 2.0f};
	std::array<float, kDim> v3{3.0f, 3.0f};
	std::array<float, kDim> v4{4.0f, 4.0f};

	cache.ensure(1, View(v1));
	cache.ensure(2, View(v2));
	cache.ensure(3, View(v3));
	cache.ensure(4, View(v4));

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

// Not a correctness oracle by itself -- primarily meant to be run under
// ThreadSanitizer to catch the exact hazard RoutingCacheHnsw's shared_mutex
// exists to prevent: hnswlib's own locking does not protect concurrent
// reads (searchKnn/getDataByLabel) against concurrent writes
// (addPoint/markDelete/resizeIndex).
TEST(RoutingCacheHnswTest, ConcurrentEnsureNearestEraseDoesNotRace) {
	constexpr int kThreads = 8;
	constexpr int kOpsPerThread = 200;

	RoutingCacheHnsw cache(kDim, /*initial_capacity=*/16, /*max_distance=*/1e-3f,
												 /*max_tombstone_ratio=*/0.3);

	std::vector<std::thread> threads;
	threads.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&cache, t] {
			for (int i = 0; i < kOpsPerThread; ++i) {
				auto id = static_cast<VectorId>(t * kOpsPerThread + i) + 1;
				std::array<float, kDim> v{static_cast<float>(id), static_cast<float>(id)};

				cache.ensure(id, View(v));
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
