#include "core/region_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "core/replacement_policy.hpp"

namespace {

using arachne::HostRegionView;
using arachne::LeaseHandle;
using arachne::PromotionCandidate;
using arachne::Region;
using arachne::RegionFootprint;
using arachne::RegionId;
using arachne::RegionManager;
using arachne::ReplacementPolicy;
using arachne::VectorId;

// Everything a RecordingReplacementPolicy has observed, kept outside the
// policy object itself (which RegionManager takes ownership of via
// unique_ptr) so a test can still inspect it after construction.
struct RecordedPolicyEvents {
	std::vector<PromotionCandidate> enqueued;
	std::vector<VectorId> evicted;
	std::vector<VectorId> touched;
};

// Records every notification it receives verbatim (no dedup/filtering of its
// own), and never offers a candidate back out (onRelocationTrigger()/
// selectNextPromotionCandidate()/selectNextEvictionCandidate() all report
// "nothing to do") -- purpose-built to observe exactly which
// ReplacementPolicy hooks RegionManager calls, when, and how many times, in
// isolation from replacement_policy_test.cpp's own FifoReplacementPolicy
// coverage of a real policy's *reaction* to the same hooks.
class RecordingReplacementPolicy : public ReplacementPolicy {
 public:
	explicit RecordingReplacementPolicy(RecordedPolicyEvents* events) : events_(events) {}

	void enqueueCandidate(PromotionCandidate candidate) override { events_->enqueued.push_back(std::move(candidate)); }
	void onAnchorEvicted(VectorId anchor_id) override { events_->evicted.push_back(anchor_id); }
	void onAnchorTouched(VectorId anchor_id) override { events_->touched.push_back(anchor_id); }
	bool onRelocationTrigger() override { return false; }
	bool hasPendingCandidates() const override { return false; }
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override { return std::nullopt; }
	std::optional<VectorId> selectNextEvictionCandidate(VectorId) override { return std::nullopt; }

 private:
	RecordedPolicyEvents* events_;
};

TEST(RegionManagerTest, RegionOfUnregisteredThrows) {
	RegionManager manager;
	EXPECT_THROW(manager.regionOf(1), std::invalid_argument);
}

TEST(RegionManagerTest, IsRegisteredReflectsRegistration) {
	RegionManager manager;
	EXPECT_FALSE(manager.isRegistered(1));

	manager.registerRegion(1, HostRegionView{});
	EXPECT_TRUE(manager.isRegistered(1));
}

TEST(RegionManagerTest, RegisterRegionRecordsHostView) {
	RegionManager manager;
	int payload = 0;
	HostRegionView host{&payload, 64};

	manager.registerRegion(1, host);

	Region region = manager.regionOf(1);
	EXPECT_EQ(region.id, 1u);
	EXPECT_EQ(region.host.ptr, &payload);
	EXPECT_EQ(region.host.bytes, 64u);
	EXPECT_FALSE(region.lease.valid());
	EXPECT_FALSE(region.device.valid());
}

TEST(RegionManagerTest, RegisterRegionIsIdempotent) {
	RegionManager manager;
	int first = 0;
	int second = 0;
	manager.registerRegion(1, HostRegionView{&first, 16});
	manager.registerRegion(1, HostRegionView{&second, 32});  // no-op: already registered

	EXPECT_EQ(manager.regionOf(1).host.ptr, &first);
	EXPECT_EQ(manager.regionOf(1).host.bytes, 16u);
}

TEST(RegionManagerTest, RegionsOfIsEmptyForUnknownAnchor) {
	RegionManager manager;
	EXPECT_TRUE(manager.regionsOf(1).empty());
}

TEST(RegionManagerTest, AddDependencyOnUnregisteredRegionFails) {
	RegionManager manager;
	EXPECT_FALSE(manager.addDependency(1, 42));
	EXPECT_TRUE(manager.regionsOf(1).empty());
}

TEST(RegionManagerTest, AddDependencyRecordsMembership) {
	RegionManager manager;
	manager.registerRegion(42, HostRegionView{});

	EXPECT_TRUE(manager.addDependency(1, 42));

	std::vector<RegionId> regions = manager.regionsOf(1);
	ASSERT_EQ(regions.size(), 1u);
	EXPECT_EQ(regions[0], 42u);
}

TEST(RegionManagerTest, AddDependencyIsIdempotent) {
	RegionManager manager;
	manager.registerRegion(42, HostRegionView{});

	manager.addDependency(1, 42);
	manager.addDependency(1, 42);

	EXPECT_EQ(manager.regionsOf(1).size(), 1u);
}

TEST(RegionManagerTest, DistinctRegionsAccumulateSeparately) {
	RegionManager manager;
	manager.registerRegion(10, HostRegionView{});
	manager.registerRegion(20, HostRegionView{});

	manager.addDependency(1, 10);
	manager.addDependency(1, 20);

	EXPECT_EQ(manager.regionsOf(1).size(), 2u);
}

TEST(RegionManagerTest, RemoveDependencyOfUnknownPairReturnsFalse) {
	RegionManager manager;
	manager.registerRegion(42, HostRegionView{});
	EXPECT_FALSE(manager.removeDependency(1, 42));
}

TEST(RegionManagerTest, RemoveDependencyReturnsTrueOnLastDependent) {
	RegionManager manager;
	manager.registerRegion(42, HostRegionView{});
	manager.addDependency(1, 42);

	EXPECT_TRUE(manager.removeDependency(1, 42));
	EXPECT_TRUE(manager.regionsOf(1).empty());
}

TEST(RegionManagerTest, RemoveDependencyReturnsFalseWhenOthersStillDepend) {
	RegionManager manager;
	manager.registerRegion(42, HostRegionView{});
	manager.addDependency(1, 42);
	manager.addDependency(2, 42);

	EXPECT_FALSE(manager.removeDependency(1, 42));  // anchor 2 still depends on it
	EXPECT_TRUE(manager.regionsOf(1).empty());
	EXPECT_EQ(manager.regionsOf(2).size(), 1u);

	EXPECT_TRUE(manager.removeDependency(2, 42));  // now the last one
}

TEST(RegionManagerTest, ForgetOfUnknownAnchorIsANoop) {
	RegionManager manager;
	EXPECT_TRUE(manager.forget(999).empty());
}

TEST(RegionManagerTest, ForgetReturnsOnlyRegionsThatDroppedToZeroDependents) {
	RegionManager manager;
	manager.registerRegion(10, HostRegionView{});  // exclusive to anchor 1
	manager.registerRegion(20, HostRegionView{});  // shared with anchor 2
	manager.addDependency(1, 10);
	manager.addDependency(1, 20);
	manager.addDependency(2, 20);

	std::vector<RegionId> orphaned = manager.forget(1);

	ASSERT_EQ(orphaned.size(), 1u);
	EXPECT_EQ(orphaned[0], 10u);
	EXPECT_TRUE(manager.regionsOf(1).empty());
	EXPECT_EQ(manager.regionsOf(2).size(), 1u);  // untouched
}

TEST(RegionManagerTest, SetLeaseAndSetDeviceUpdateRegionOf) {
	RegionManager manager;
	manager.registerRegion(1, HostRegionView{});

	manager.setLease(1, LeaseHandle{1, 7});
	manager.setDevice(1, arachne::gpu::DeviceRegionHandle{99});

	Region region = manager.regionOf(1);
	EXPECT_TRUE(region.lease.valid());
	EXPECT_EQ(region.lease.epoch, 7u);
	EXPECT_TRUE(region.device.valid());
	EXPECT_EQ(region.device.id, 99u);
}

TEST(RegionManagerTest, ClearResidencyResetsLeaseAndDeviceButKeepsRegistration) {
	RegionManager manager;
	manager.registerRegion(1, HostRegionView{});
	manager.setLease(1, LeaseHandle{1, 7});
	manager.setDevice(1, arachne::gpu::DeviceRegionHandle{99});

	manager.clearResidency(1);

	EXPECT_TRUE(manager.isRegistered(1));
	Region region = manager.regionOf(1);
	EXPECT_FALSE(region.lease.valid());
	EXPECT_FALSE(region.device.valid());
}

TEST(RegionManagerTest, SharedRegionExposesTheSameLeaseToEveryDependent) {
	// The whole point of merging Stitch into Region: two Anchors depending on
	// the same Region see the exact same lease, rather than each holding an
	// independent copy negotiated separately.
	RegionManager manager;
	manager.registerRegion(42, HostRegionView{});
	manager.addDependency(1, 42);
	manager.addDependency(2, 42);

	manager.setLease(42, LeaseHandle{42, 5});

	EXPECT_EQ(manager.regionOf(42).lease.epoch, 5u);
	// Both anchors resolve the same RegionId to the identical Region record.
	EXPECT_EQ(manager.regionsOf(1)[0], manager.regionsOf(2)[0]);
}

// ---------------------------------------------------------------------------
// recordTraversal(): the traverse-level hotness signal (see
// core/replacement_policy.hpp's onAnchorTouched() doc comment) -- exercised
// directly against RegionManager's dependency graph, with no Coordinator/
// adapter/GPU involved, since fan-out from touched Regions to dependent
// Anchors is pure bookkeeping.
// ---------------------------------------------------------------------------

TEST(RegionManagerTest, RecordTraversalNotifiesEveryDependentAnchorOfEachTouchedRegion) {
	RecordedPolicyEvents events;
	RegionManager manager(std::make_unique<RecordingReplacementPolicy>(&events));
	manager.registerRegion(10, HostRegionView{});
	manager.registerRegion(20, HostRegionView{});
	manager.addDependency(1, 10);
	manager.addDependency(2, 20);

	manager.recordTraversal(RegionFootprint{{10, 20}});

	std::sort(events.touched.begin(), events.touched.end());
	ASSERT_EQ(events.touched.size(), 2u);
	EXPECT_EQ(events.touched[0], 1u);
	EXPECT_EQ(events.touched[1], 2u);
}

TEST(RegionManagerTest, RecordTraversalDedupesAnAnchorDependingOnMultipleTouchedRegions) {
	RecordedPolicyEvents events;
	RegionManager manager(std::make_unique<RecordingReplacementPolicy>(&events));
	manager.registerRegion(10, HostRegionView{});
	manager.registerRegion(20, HostRegionView{});
	manager.addDependency(1, 10);
	manager.addDependency(1, 20);  // anchor 1 depends on both touched regions

	manager.recordTraversal(RegionFootprint{{10, 20}});

	ASSERT_EQ(events.touched.size(), 1u);  // reported once per recordTraversal() call, not once per region
	EXPECT_EQ(events.touched[0], 1u);
}

TEST(RegionManagerTest, RecordTraversalOfUnregisteredRegionIsANoop) {
	RecordedPolicyEvents events;
	RegionManager manager(std::make_unique<RecordingReplacementPolicy>(&events));

	EXPECT_NO_THROW(manager.recordTraversal(RegionFootprint{{999}}));  // never registered
	EXPECT_TRUE(events.touched.empty());
}

TEST(RegionManagerTest, RecordTraversalOfOrphanedRegionIsANoop) {
	RecordedPolicyEvents events;
	RegionManager manager(std::make_unique<RecordingReplacementPolicy>(&events));
	manager.registerRegion(10, HostRegionView{});  // registered, but nobody depends on it

	manager.recordTraversal(RegionFootprint{{10}});
	EXPECT_TRUE(events.touched.empty());
}

TEST(RegionManagerTest, RecordTraversalOfEmptyFootprintIsANoop) {
	RecordedPolicyEvents events;
	RegionManager manager(std::make_unique<RecordingReplacementPolicy>(&events));
	manager.registerRegion(10, HostRegionView{});
	manager.addDependency(1, 10);

	manager.recordTraversal(RegionFootprint{});
	EXPECT_TRUE(events.touched.empty());
}

// ---------------------------------------------------------------------------
// requestPromotion(): a pure enqueue onto RegionManager's own MPSC intake
// queue (pending_promotions_) -- must *not* touch replacement_policy_ at all
// on the calling thread. Only the Coordinator thread (started via start())
// ever drains that queue and hands candidates to the policy via
// enqueueCandidate() -- see the class doc comment (region_manager.hpp) and
// ReplacementPolicy::enqueueCandidate()'s own doc comment for why. This test
// never calls start(), so if requestPromotion() ever touched the policy
// directly, events.enqueued would already be non-empty here.
// ---------------------------------------------------------------------------

TEST(RegionManagerTest, RequestPromotionDoesNotTouchReplacementPolicyDirectly) {
	RecordedPolicyEvents events;
	RegionManager manager(std::make_unique<RecordingReplacementPolicy>(&events));
	manager.registerRegion(1, HostRegionView{});

	manager.requestPromotion(100, RegionFootprint{{1}});

	EXPECT_TRUE(events.enqueued.empty());
	EXPECT_TRUE(events.evicted.empty());
	EXPECT_TRUE(events.touched.empty());
}

// ---------------------------------------------------------------------------
// Stress tests: many threads hammering the same RegionManager concurrently.
// The primary thing these prove is that the mutex-guarded bookkeeping
// doesn't deadlock or corrupt itself under heavy concurrent load; the first
// test additionally proves a specific correctness property (refcounting):
// exactly one of many racing "last dependent leaves" events may ever fire
// for a given Region.
// ---------------------------------------------------------------------------

TEST(RegionManagerStressTest, ConcurrentDependencyChurnReportsExactlyOneLastDependentEvent) {
	RegionManager manager;
	constexpr RegionId kRegion = 1;
	manager.registerRegion(kRegion, HostRegionView{});

	constexpr int kThreads = 64;
	constexpr int kChurnRoundsPerThread = 500;

	// Phase 1: each thread repeatedly adds/removes its own (distinct) Anchor
	// as a dependent of the same shared Region, purely to stress the lock --
	// the return values here are not asserted on, since with many other
	// threads concurrently doing the same thing, whether *this* thread's
	// removeDependency() happens to be the last one at that instant is
	// nondeterministic. Ends with every thread depending on kRegion exactly
	// once (guaranteed by joining before phase 2 starts).
	std::vector<std::thread> threads;
	threads.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&, t] {
			VectorId anchor_id = static_cast<VectorId>(t + 1);
			for (int round = 0; round < kChurnRoundsPerThread; ++round) {
				manager.addDependency(anchor_id, kRegion);
				manager.removeDependency(anchor_id, kRegion);
			}
			ASSERT_TRUE(manager.addDependency(anchor_id, kRegion));
		});
	}
	for (std::thread& th : threads) th.join();
	threads.clear();

	// Phase 2: all kThreads Anchors now depend on kRegion. Drop them all
	// concurrently via forget() -- exactly one of these T racing calls must
	// observe "I was the last dependent" (RegionManager's own mutex
	// serializes the actual transition, however unpredictable the thread
	// scheduling is).
	std::atomic<int> last_dependent_events{0};
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&, t] {
			VectorId anchor_id = static_cast<VectorId>(t + 1);
			std::vector<RegionId> orphaned = manager.forget(anchor_id);
			if (!orphaned.empty()) {
				EXPECT_EQ(orphaned.size(), 1u);
				EXPECT_EQ(orphaned[0], kRegion);
				last_dependent_events.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	for (std::thread& th : threads) th.join();

	EXPECT_EQ(last_dependent_events.load(), 1);
	EXPECT_TRUE(manager.isRegistered(kRegion));  // registration itself outlives residency
	for (int t = 0; t < kThreads; ++t) {
		EXPECT_TRUE(manager.regionsOf(static_cast<VectorId>(t + 1)).empty());
	}
}

TEST(RegionManagerStressTest, ConcurrentRegisterAndDependencyChurnAcrossManyRegionsAndAnchors) {
	RegionManager manager;
	constexpr int kRegions = 200;
	constexpr int kAnchors = 64;
	constexpr int kThreads = 16;
	constexpr int kOpsPerThread = 5000;

	for (int r = 0; r < kRegions; ++r) {
		manager.registerRegion(static_cast<RegionId>(r + 1), HostRegionView{});
	}

	std::vector<std::thread> threads;
	threads.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&, t] {
			std::mt19937 rng(1000u + static_cast<unsigned>(t));
			std::uniform_int_distribution<int> anchor_dist(1, kAnchors);
			std::uniform_int_distribution<int> region_dist(1, kRegions);
			std::uniform_int_distribution<int> op_dist(0, 1);
			for (int i = 0; i < kOpsPerThread; ++i) {
				VectorId anchor_id = static_cast<VectorId>(anchor_dist(rng));
				RegionId region_id = static_cast<RegionId>(region_dist(rng));
				if (op_dist(rng) == 0) {
					manager.addDependency(anchor_id, region_id);
				} else {
					manager.removeDependency(anchor_id, region_id);
				}
			}
		});
	}
	for (std::thread& th : threads) th.join();

	// No crash/deadlock above is the primary thing this proves; also
	// sanity-check every surviving dependency still points at a registered,
	// in-range Region.
	for (int a = 1; a <= kAnchors; ++a) {
		for (RegionId region_id : manager.regionsOf(static_cast<VectorId>(a))) {
			EXPECT_TRUE(manager.isRegistered(region_id));
			EXPECT_GE(region_id, 1u);
			EXPECT_LE(region_id, static_cast<RegionId>(kRegions));
		}
	}

	// Clean up: every Anchor forgetting everything must not throw/crash, and
	// afterward every Anchor has zero dependencies left.
	for (int a = 1; a <= kAnchors; ++a) {
		manager.forget(static_cast<VectorId>(a));
		EXPECT_TRUE(manager.regionsOf(static_cast<VectorId>(a)).empty());
	}
}

}  // namespace
