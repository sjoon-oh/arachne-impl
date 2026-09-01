// Module-level tests for RegionManager's Coordinator: the dedicated
// background thread (see region_manager.hpp's class doc comment) that now
// owns GPU residency policy -- promotion, capacity-driven eviction,
// write-back -- lazily and batched, off of any caller's own thread. These
// tests exercise RegionManager directly (no Controller involved) so a
// failure here points specifically at the Coordinator's own lifecycle/
// lazy-trigger/capacity logic, distinct from controller_test.cpp's
// end-to-end (Controller + RegionManager) coverage of the same machinery.

#include "core/region_manager.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "adapter/region.hpp"
#include "core/routing_cache.hpp"
#include "gpu/device_context.hpp"
#include "gpu/device_region_pool.hpp"
#include "types.hpp"

namespace {

using namespace arachne;

// ---------------------------------------------------------------------------
// Test doubles -- deliberately minimal; these tests only exercise
// RegionManager's own promotion/eviction bookkeeping, never Controller's
// routing/scheduling, so traverseHost()/modifyHost() are never called.
// ---------------------------------------------------------------------------

class FakeRegion : public IRegion {
 public:
	FakeRegion() = default;
	FakeRegion(RegionId id, HostRegionView host) : id_(id), host_(host) {}

	RegionId id() const override { return id_; }
	RegionFootprint footprint() const override { return RegionFootprint{{id_}}; }
	HostRegionView hostView() const override { return host_; }
	LeaseHandle acquireWriteLease() override { return LeaseHandle{id_, ++epoch_}; }
	void releaseWriteLease(LeaseHandle) override {}
	void applyLocalModification(LeaseHandle, const ModificationDelta&) override {}
	ReconciliationReport reconcileBoundary() override { return ReconciliationReport{}; }

 private:
	RegionId id_ = 0;
	HostRegionView host_;
	std::uint64_t epoch_ = 0;
};

class FakeAdapter : public IAdapter {
 public:
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		return std::vector<TraverseResult>(requests.size());
	}
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		return std::vector<ModifyResult>(requests.size());
	}

	IRegion* resolveRegion(RegionId id) override {
		auto it = regions_.find(id);
		return it == regions_.end() ? nullptr : &it->second;
	}
	std::vector<RegionId> allRegions() const override {
		std::vector<RegionId> ids;
		ids.reserve(regions_.size());
		for (const auto& [id, region] : regions_) ids.push_back(id);
		return ids;
	}

	void addRegion(RegionId id, HostRegionView host) { regions_.emplace(id, FakeRegion(id, host)); }

 private:
	std::unordered_map<RegionId, FakeRegion> regions_;
};

// Records every ensure()/erase() call verbatim so tests can assert exactly
// when RegionManager registers/removes an Anchor -- now tied to actual
// promotion-grant/eviction time (see region_manager.hpp), not to every
// query. nearest() always misses; these tests never exercise routing itself.
class FakeRoutingCache : public RoutingCache {
 public:
	FakeRoutingCache() : RoutingCache(/*dim=*/1, DistanceMetric::L2, VectorDType::Float32) {}

	std::optional<VectorId> nearestImpl(const VectorView&) override { return std::nullopt; }
	VectorId ensure(VectorId id, const VectorView& vector, float) override {
		ensured.push_back(id);
		const auto* bytes = static_cast<const std::byte*>(vector.data);
		last_ensure_bytes.assign(bytes, bytes + static_cast<std::size_t>(vector.dim) * VectorElementSize(vector.dtype));
		return id;
	}
	void erase(VectorId id) override { erased.push_back(id); }

	std::vector<VectorId> ensured;
	std::vector<VectorId> erased;
	std::vector<std::byte> last_ensure_bytes;
};

class IntakeObservingPolicy : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override {
		std::lock_guard<std::mutex> lock(mutex_);
		pending_.push_back(std::move(candidate));
		cv_.notify_all();
	}
	void onAnchorEvicted(VectorId) override {}
	void onAnchorTouched(VectorId) override {}
	bool onRelocationTrigger() override { return false; }
	bool hasPendingCandidates() const override {
		std::lock_guard<std::mutex> lock(mutex_);
		return !pending_.empty();
	}
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override {
		std::lock_guard<std::mutex> lock(mutex_);
		if (pending_.empty()) return std::nullopt;
		PromotionCandidate candidate = std::move(pending_.front());
		pending_.pop_front();
		return candidate;
	}
	std::optional<VectorId> selectEvictionCandidate(VectorId, std::size_t, const std::vector<EvictionCandidate>&,
			const std::unordered_set<VectorId>& = {}) override {
		return std::nullopt;
	}
	bool waitForIntake(std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> lock(mutex_);
		return cv_.wait_for(lock, timeout, [this] { return !pending_.empty(); });
	}

 private:
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::deque<PromotionCandidate> pending_;
};

// A generous interval used whenever a test wants to prove something is
// *not* processed until waitIdle() forces it -- long enough that the
// periodic tick essentially never wins the race against the assertion that
// immediately follows.
constexpr std::chrono::milliseconds kLongInterval{200};
// A short interval used whenever a test wants to prove the periodic tick
// alone (no waitIdle()) eventually processes pending work.
constexpr std::chrono::milliseconds kShortInterval{5};

// ---------------------------------------------------------------------------

TEST(RegionManagerCoordinatorTest, ShutdownWithoutStartIsANoop) {
	RegionManager manager;
	EXPECT_NO_THROW(manager.shutdown());
}

TEST(RegionManagerCoordinatorTest, StartTwiceThrows) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	gpu::DeviceContext device;
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;

	manager.start(adapter, pool, routing_cache);
	EXPECT_THROW(manager.start(adapter, pool, routing_cache), std::logic_error);

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, RequestPromotionIsLazyUntilWaitIdle) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(100, RegionFootprint{{1}});
	EXPECT_FALSE(manager.regionOf(1).device.valid());  // enqueued, not yet processed

	manager.waitIdle();
	EXPECT_TRUE(manager.regionOf(1).device.valid());
	EXPECT_EQ(manager.regionsOf(100).size(), 1u);
	EXPECT_EQ(manager.stats().regions_promoted_total, 1u);
	// Registered in RoutingCache now that the promotion actually landed --
	// see region_manager.hpp's class doc comment.
	EXPECT_EQ(routing_cache.ensured, std::vector<VectorId>{100});

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, CandidateIntakeIsImmediateButExecutionWaitsForDeadline) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	auto policy = std::make_unique<IntakeObservingPolicy>();
	IntakeObservingPolicy* observed = policy.get();
	RegionManager manager(std::move(policy));
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{std::chrono::seconds(2)});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);
	manager.requestPromotion(100, RegionFootprint{{1}});

	EXPECT_TRUE(observed->waitForIntake(std::chrono::milliseconds(500)));
	EXPECT_FALSE(manager.regionOf(1).device.valid());
	manager.waitIdle();
	EXPECT_TRUE(manager.regionOf(1).device.valid());
	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, ReleaseAnchorBookkeepingIsImmediateButReclaimIsLazy) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(100, RegionFootprint{{1}});
	manager.waitIdle();
	ASSERT_TRUE(manager.regionOf(1).device.valid());
	ASSERT_EQ(manager.regionsOf(100).size(), 1u);

	manager.releaseAnchor(100);
	// Dependency-graph bookkeeping, the Resident -> Retiring transition, and
	// RoutingCache erasure are synchronous. The device remains valid until
	// the Coordinator observes zero logical pins and completes reclaim.
	EXPECT_TRUE(manager.regionsOf(100).empty());
	EXPECT_EQ(manager.regionOf(1).residency_state, RegionResidencyState::Retiring);
	EXPECT_TRUE(manager.regionOf(1).device.valid());
	EXPECT_EQ(routing_cache.erased, std::vector<VectorId>{100});

	// The actual GPU free() is deferred -- force it and check it landed.
	manager.waitIdle();
	EXPECT_EQ(manager.stats().anchor_evictions_total, 1u);
	EXPECT_EQ(manager.regionOf(1).residency_state, RegionResidencyState::HostOnly);
	EXPECT_FALSE(manager.regionOf(1).device.valid());
	EXPECT_EQ(pool.bytesAllocated(), 0u);

	manager.shutdown();
}

// isKnownAnchor() exists purely for Controller::MintAnchorId()'s wraparound-
// collision guard (controller.cpp) -- these three cover the three states an
// Anchor id can be in relative to it: never seen, currently resident, and
// released-but-permanently-remembered. The guard's own retry-on-collision
// loop isn't separately exercised here -- doing so would mean actually
// driving a 64-bit counter to its limit, which isn't practical; this is the
// primitive it's built on, tested directly instead.
TEST(RegionManagerCoordinatorTest, IsKnownAnchorIsFalseForAnIdThatWasNeverAssigned) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	gpu::DeviceContext device;
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	EXPECT_FALSE(manager.isKnownAnchor(999));

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, IsKnownAnchorIsTrueWhileAnAnchorCurrentlyDependsOnARegion) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	EXPECT_FALSE(manager.isKnownAnchor(100)) << "not yet requested -- must not already read as known";
	manager.requestPromotion(100, RegionFootprint{{1}});
	manager.waitIdle();
	EXPECT_TRUE(manager.isKnownAnchor(100));

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, IsKnownAnchorStaysTrueAfterReleaseEvenThoughNoLongerResident) {
	// The property Controller::MintAnchorId()'s wraparound guard actually
	// depends on: an Anchor id must never look "free" again just because
	// it's no longer resident -- anchor_epoch_'s entries are never removed
	// (see that member's own doc comment) specifically so this stays true
	// forever, not just while the Anchor is still alive.
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(100, RegionFootprint{{1}});
	manager.waitIdle();
	ASSERT_TRUE(manager.isKnownAnchor(100));

	manager.releaseAnchor(100);
	ASSERT_TRUE(manager.regionsOf(100).empty()) << "no longer depends on anything -- the dependencies_ half is gone";
	EXPECT_TRUE(manager.isKnownAnchor(100)) << "must still read as known via anchor_epoch_";

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, TimelyTriggerProcessesWithoutExplicitWaitIdle) {
	// "Timely event triggering": the Coordinator's own periodic tick, with no
	// caller ever calling waitIdle(), must eventually catch up on its own.
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(100, RegionFootprint{{1}});
	std::this_thread::sleep_for(kShortInterval * 20);  // many ticks' worth of margin

	EXPECT_TRUE(manager.regionOf(1).device.valid());

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, RequestPromotionForUnregisteredRegionDoesNotCrash) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	gpu::DeviceContext device;
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	manager.requestPromotion(100, RegionFootprint{{999}});  // 999 was never registered
	manager.waitIdle();

	EXPECT_TRUE(manager.regionsOf(100).empty());
	EXPECT_EQ(manager.stats().regions_promoted_total, 0u);
	// Never gained any residency -- nothing for RoutingCache to register.
	EXPECT_TRUE(routing_cache.ensured.empty());

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, CapacityPressureEvictsOldestViaReplacementPolicy) {
	// Mirrors controller_test.cpp's PromoteEvictsMultipleVictimsWhenOneIsNotEnoughCapacity,
	// but drives RegionManager directly -- proving the capacity-retry/eviction
	// loop is correct as RegionManager's own responsibility, independent of
	// Controller.
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 2 * kBytes,
														 gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	std::vector<std::vector<std::byte>> buffers(3, std::vector<std::byte>(kBytes, std::byte{0}));
	for (RegionId id = 1; id <= 3; ++id) {
		HostRegionView host{buffers[id - 1].data(), kBytes, 0};
		adapter.addRegion(id, host);
		manager.registerRegion(id, host);
	}

	manager.requestPromotion(101, RegionFootprint{{1}});
	manager.waitIdle();
	manager.requestPromotion(102, RegionFootprint{{2}});
	manager.waitIdle();  // budget now full (2 regions of 256 bytes each)
	const std::uint64_t oldest_handle = manager.regionOf(1).device.id;

	manager.requestPromotion(103, RegionFootprint{{3}});
	manager.waitIdle();  // needs capacity -- evicts anchor 101 (FIFO-oldest)

	EXPECT_FALSE(manager.regionOf(1).device.valid());  // evicted
	EXPECT_TRUE(manager.regionOf(2).device.valid());   // untouched
	EXPECT_TRUE(manager.regionOf(3).device.valid());   // newly promoted
	EXPECT_EQ(manager.regionOf(3).device.id, oldest_handle);  // near-fit allocation reuse
	EXPECT_EQ(manager.stats().anchor_evictions_total, 1u);
	EXPECT_LE(manager.stats().gpu_bytes_allocated, 2 * kBytes);
	// Capacity-driven eviction (evictAnchorNow(), not releaseAnchor()) must
	// also erase from RoutingCache -- anchor 101 no longer represents live
	// GPU residency.
	EXPECT_EQ(routing_cache.erased, std::vector<VectorId>{101});
	EXPECT_EQ(routing_cache.ensured, (std::vector<VectorId>{101, 102, 103}));
	EXPECT_EQ(manager.stats().near_fit_reuses_total, 1u);

	manager.shutdown();
}

// Two Anchors sharing the exact same Region -- with the default
// max_eviction_group_size=1, that Region becomes permanently unreclaimable
// the moment a second Anchor depends on it (dependents_[region].size() == 2,
// so neither Anchor's own EvictionCandidate::reclaimable_bytes ever counts
// it -- see buildEvictionCandidates()'s doc comment). This is exactly what a
// heavily fan-in "hub" Region hits in a read-heavy production workload.
// Raising max_eviction_group_size (and configuring a lenient
// admission_hysteresis so this test's outcome doesn't hinge on a
// close-to-tied heat comparison between two Anchors promoted milliseconds
// apart) lets the two Anchors be grouped together, so the shared Region
// becomes reclaimable by evicting the whole group jointly.
TEST(RegionManagerCoordinatorTest, GroupEvictionReclaimsRegionSharedByMultipleAnchorsWhenCapAllowsIt) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes,
														 gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	CostAwareReplacementConfig policy_config;
	policy_config.admission_hysteresis = 0.1;  // comfortably admit once *any* victim is found
	RegionManager manager(std::make_unique<CostAwareReplacementPolicy>(policy_config));
	CoordinatorConfig config{kShortInterval};
	config.group_merge_overlap_threshold = 0.5;
	config.max_eviction_group_size = 2;
	manager.start(adapter, pool, routing_cache, config);

	std::vector<std::vector<std::byte>> buffers(2, std::vector<std::byte>(kBytes, std::byte{0}));
	for (RegionId id = 1; id <= 2; ++id) {
		HostRegionView host{buffers[id - 1].data(), kBytes, 0};
		adapter.addRegion(id, host);
		manager.registerRegion(id, host);
	}

	manager.requestPromotion(201, RegionFootprint{{1}});
	manager.waitIdle();
	manager.requestPromotion(202, RegionFootprint{{1}});  // same Region, already resident -- free to join
	manager.waitIdle();
	ASSERT_TRUE(manager.regionOf(1).device.valid());
	ASSERT_FALSE(manager.regionOf(2).device.valid());

	// Budget (256B) is already full with Region 1 alone; promoting 203's
	// Region 2 needs eviction. Region 1 has two dependents (201, 202) -- under
	// the default sole-ownership rule this would have zero eligible victims.
	manager.requestPromotion(203, RegionFootprint{{2}});
	manager.waitIdle();

	EXPECT_FALSE(manager.regionOf(1).device.valid());  // whole group evicted together
	EXPECT_TRUE(manager.regionOf(2).device.valid());   // newly promoted
	EXPECT_TRUE(manager.regionsOf(201).empty());
	EXPECT_TRUE(manager.regionsOf(202).empty());
	EXPECT_GE(manager.stats().anchor_evictions_total, 2u);  // both group members evicted
	EXPECT_NE(std::find(routing_cache.erased.begin(), routing_cache.erased.end(), 201), routing_cache.erased.end());
	EXPECT_NE(std::find(routing_cache.erased.begin(), routing_cache.erased.end(), 202), routing_cache.erased.end());

	manager.shutdown();
}

// Same setup as above, but with the *default* max_eviction_group_size (1) --
// every Anchor is always its own singleton group, so a Region with more than
// one dependent is never reclaimable, exactly reproducing this port's
// original sole-ownership-only rule. Confirms the new grouping machinery is
// fully opt-in: leaving CoordinatorConfig untouched changes nothing.
TEST(RegionManagerCoordinatorTest, SharedRegionStaysUnreclaimableWithDefaultGroupCap) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes,
														 gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<CostAwareReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});  // group cap defaults to 1

	std::vector<std::vector<std::byte>> buffers(2, std::vector<std::byte>(kBytes, std::byte{0}));
	for (RegionId id = 1; id <= 2; ++id) {
		HostRegionView host{buffers[id - 1].data(), kBytes, 0};
		adapter.addRegion(id, host);
		manager.registerRegion(id, host);
	}

	manager.requestPromotion(301, RegionFootprint{{1}});
	manager.waitIdle();
	manager.requestPromotion(302, RegionFootprint{{1}});
	manager.waitIdle();
	ASSERT_TRUE(manager.regionOf(1).device.valid());

	manager.requestPromotion(303, RegionFootprint{{2}});
	manager.waitIdle();

	EXPECT_TRUE(manager.regionOf(1).device.valid());   // still shared by 301+302, never reclaimed
	EXPECT_FALSE(manager.regionOf(2).device.valid());  // 303 never got a chance to promote
	EXPECT_FALSE(manager.regionsOf(301).empty());
	EXPECT_FALSE(manager.regionsOf(302).empty());
	EXPECT_TRUE(manager.regionsOf(303).empty());
	EXPECT_GE(manager.stats().candidates_rejected_total, 1u);

	manager.shutdown();
}

// Direct RegionManager-level reproduction of the requeue bug this session's
// fix (buildRelocationPlan(), region_manager.cpp) addresses: four promotions
// submitted back-to-back with no intermediate waitIdle() all land in the
// Coordinator's very first (forced) pass together. 101/102/103 (256B each)
// fill the 768B budget exactly; 104 (512B) can't join that same pass. Before
// the fix, 104 was dropped permanently instead of requeued, because
// forced/stop drains suppressed retry unconditionally, regardless of whether
// retrying was actually safe. See buildRelocationPlan()'s own comment for
// why requeuing it here is provably safe (a bounded number of passes, since
// plan.promotions is non-empty at drop time -- 101/102/103 already claimed
// room in this exact pass) even though this whole thing runs inside one
// forced waitIdle() call.
TEST(RegionManagerCoordinatorTest, SameBatchOverflowRequeuesWithoutIntermediateWaitIdle) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kSmallBytes = 256;
	constexpr std::size_t kBigBytes = 512;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 3 * kSmallBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kSmallBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> data_a(kSmallBytes, std::byte{0xA});
	std::vector<std::byte> data_b(kSmallBytes, std::byte{0xB});
	std::vector<std::byte> data_c(kSmallBytes, std::byte{0xC});
	std::vector<std::byte> data_d(kBigBytes, std::byte{0xD});
	HostRegionView host_a{data_a.data(), kSmallBytes, 0};
	HostRegionView host_b{data_b.data(), kSmallBytes, 0};
	HostRegionView host_c{data_c.data(), kSmallBytes, 0};
	HostRegionView host_d{data_d.data(), kBigBytes, 0};
	adapter.addRegion(1, host_a);
	adapter.addRegion(2, host_b);
	adapter.addRegion(3, host_c);
	adapter.addRegion(4, host_d);
	manager.registerRegion(1, host_a);
	manager.registerRegion(2, host_b);
	manager.registerRegion(3, host_c);
	manager.registerRegion(4, host_d);

	manager.requestPromotion(101, RegionFootprint{{1}});
	manager.requestPromotion(102, RegionFootprint{{2}});
	manager.requestPromotion(103, RegionFootprint{{3}});
	manager.requestPromotion(104, RegionFootprint{{4}});
	// Deliberately no waitIdle() between requests -- all four sit pending
	// before the Coordinator ever wakes up.

	manager.waitIdle();

	EXPECT_FALSE(manager.regionOf(1).device.valid());  // evicted to make room for 104
	EXPECT_FALSE(manager.regionOf(2).device.valid());  // evicted to make room for 104
	EXPECT_TRUE(manager.regionOf(3).device.valid());   // untouched -- one eviction wasn't enough, two was
	EXPECT_TRUE(manager.regionOf(4).device.valid());   // eventually promoted, not lost
	EXPECT_GT(manager.stats().relocation_batches_total, 1u)
			<< "104 should have needed a second Coordinator pass within this one waitIdle() call";
	EXPECT_GE(manager.stats().candidates_requeued_total, 1u)
			<< "104 should have been requeued at least once instead of dropped permanently";

	manager.shutdown();
}

// Termination-safety companion to the test above: a candidate that cannot
// fit even *alone* (bigger than the entire budget, not just crowded out by
// same-pass siblings) must still be dropped for good on its very first
// standalone attempt, never requeued -- otherwise a forced drain
// (waitIdle()/shutdown()) could spin forever requeuing something that will
// never succeed. See buildRelocationPlan()'s own comment for why
// plan.promotions being empty at drop time is exactly the condition that
// distinguishes this permanent case from the requeueable one above.
TEST(RegionManagerCoordinatorTest, SoloOversizedCandidateIsDroppedPermanentlyNotRequeued) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kSmallBytes = 256;
	constexpr std::size_t kHugeBytes = 4 * kSmallBytes;  // bigger than the whole budget below
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 3 * kSmallBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kSmallBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> data_huge(kHugeBytes, std::byte{0xE});
	HostRegionView host_huge{data_huge.data(), kHugeBytes, 0};
	adapter.addRegion(1, host_huge);
	manager.registerRegion(1, host_huge);

	manager.requestPromotion(201, RegionFootprint{{1}});
	manager.waitIdle();  // must return promptly -- not hang retrying forever

	EXPECT_FALSE(manager.regionOf(1).device.valid());  // never fits, dropped for good
	EXPECT_TRUE(manager.regionsOf(201).empty());
	EXPECT_EQ(manager.stats().candidates_requeued_total, 0u)
			<< "a candidate bigger than the whole budget must never be requeued";

	manager.shutdown();
}

// max_promotion_bytes_per_pass caps how much one Coordinator pass promotes
// (see CoordinatorConfig's own doc comment) -- proven here by giving the
// Coordinator four individually-fitting candidates (the budget below fits
// all four with no eviction needed at all) but a cap tight enough that only
// one can join any single pass, then checking relocation_batches_total/
// candidates_requeued_total reflect that one waitIdle() call actually took
// multiple passes to drain, not one. Depends on the fix above -- before it,
// the cap already existed but bumped candidates were dropped instead of
// requeued during this forced waitIdle() drain.
TEST(RegionManagerCoordinatorTest, MaxPromotionBytesPerPassSplitsOneWaitIdleIntoMultiplePasses) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kSmallBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 4 * kSmallBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kSmallBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	CoordinatorConfig config{kLongInterval};
	config.max_promotion_bytes_per_pass = kSmallBytes;  // exactly one region's worth per pass
	manager.start(adapter, pool, routing_cache, config);

	std::vector<std::byte> data_1(kSmallBytes, std::byte{1});
	std::vector<std::byte> data_2(kSmallBytes, std::byte{2});
	std::vector<std::byte> data_3(kSmallBytes, std::byte{3});
	std::vector<std::byte> data_4(kSmallBytes, std::byte{4});
	HostRegionView host_1{data_1.data(), kSmallBytes, 0};
	HostRegionView host_2{data_2.data(), kSmallBytes, 0};
	HostRegionView host_3{data_3.data(), kSmallBytes, 0};
	HostRegionView host_4{data_4.data(), kSmallBytes, 0};
	adapter.addRegion(1, host_1);
	adapter.addRegion(2, host_2);
	adapter.addRegion(3, host_3);
	adapter.addRegion(4, host_4);
	manager.registerRegion(1, host_1);
	manager.registerRegion(2, host_2);
	manager.registerRegion(3, host_3);
	manager.registerRegion(4, host_4);

	manager.requestPromotion(301, RegionFootprint{{1}});
	manager.requestPromotion(302, RegionFootprint{{2}});
	manager.requestPromotion(303, RegionFootprint{{3}});
	manager.requestPromotion(304, RegionFootprint{{4}});
	// All 4 submitted before any waitIdle() -- budget (4*256=1024) already
	// fits all of them with zero eviction, so this is purely about how many
	// passes the promotion-side cap forces within one waitIdle() call, not
	// about capacity pressure.
	manager.waitIdle();

	EXPECT_TRUE(manager.regionOf(1).device.valid());
	EXPECT_TRUE(manager.regionOf(2).device.valid());
	EXPECT_TRUE(manager.regionOf(3).device.valid());
	EXPECT_TRUE(manager.regionOf(4).device.valid());
	// Exact counts (not just "more than one"): each pass admits its first
	// candidate unconditionally, then defers everything else that would push
	// this pass over the 256B cap -- so 4 candidates, capped at 1 per pass,
	// take exactly 4 passes, requeuing each of the 3 non-first candidates
	// exactly once.
	EXPECT_EQ(manager.stats().relocation_batches_total, 4u);
	EXPECT_EQ(manager.stats().candidates_requeued_total, 3u);

	manager.shutdown();
}

// Records the raw pointer behind AdmissionContext::eviction_candidates for
// every candidate it's asked about, then always Rejects -- deliberately
// never admits anything, so buildRelocationPlan_collect's while(true) loop
// keeps pulling until the whole batch is drained in one single pass,
// without needing an actual working promotion/eviction pipeline plugged in.
class RecordingAlwaysRejectPolicy : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override {
		std::lock_guard<std::mutex> lock(mutex_);
		pending_.push_back(std::move(candidate));
	}
	void onAnchorEvicted(VectorId) override {}
	void onAnchorTouched(VectorId) override {}
	bool onRelocationTrigger() override {
		std::lock_guard<std::mutex> lock(mutex_);
		return !pending_.empty();
	}
	bool hasPendingCandidates() const override {
		std::lock_guard<std::mutex> lock(mutex_);
		return !pending_.empty();
	}
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override {
		std::lock_guard<std::mutex> lock(mutex_);
		if (pending_.empty()) return std::nullopt;
		PromotionCandidate candidate = std::move(pending_.front());
		pending_.pop_front();
		return candidate;
	}
	std::optional<VectorId> selectEvictionCandidate(VectorId, std::size_t, const std::vector<EvictionCandidate>&,
			const std::unordered_set<VectorId>& = {}) override {
		return std::nullopt;
	}
	BatchAdmissionDecision evaluateBatchAdmission(const PromotionCandidate&, const AdmissionContext& admission,
			const RelocationBatchContext&) override {
		std::lock_guard<std::mutex> lock(mutex_);
		observed_pointers.push_back(admission.eviction_candidates.get());
		return BatchAdmissionDecision::Reject;
	}

	mutable std::mutex mutex_;
	mutable std::vector<const std::vector<EvictionCandidate>*> observed_pointers;

 private:
	std::deque<PromotionCandidate> pending_;
};

// Regression test for AdmissionContext::eviction_candidates being a
// shared_ptr into RegionManager's own pass-local cache rather than an owned
// std::vector -- see that field's own doc comment (replacement_policy.hpp)
// for the deep-copy cost this replaced, measured to run into minutes at
// 1M-vector scale (cpp/test/index/report/, the buildRelocationPlan_collect
// finding). Proves the *sharing*, not just the values: every candidate
// examined within one pass must see the exact same underlying snapshot
// (same address), not merely an equal one.
TEST(RegionManagerCoordinatorTest, AdmissionContextSharesTheSameEvictionCandidatesSnapshotAcrossOnePass) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kUnitBytes = 256;
	// Region payload is 2 allocation units; budget is exactly 1 (already a
	// multiple of kUnitBytes, so Pooled's own unit-rounding of the *budget*
	// is a no-op here -- unlike a budget smaller than one whole unit, which
	// Pooled would silently round up to fill it, defeating this setup; see
	// the metadata pool note below for the other unit-rounding trap this
	// avoids). So `available` (1 unit) is below every candidate's
	// incremental_bytes (2 units) regardless of whether anything's ever
	// actually promoted (this policy never admits), and
	// buildAdmissionContext() takes the eviction_candidates branch for every
	// one of them, every time -- exactly the condition this test needs.
	constexpr std::size_t kRegionBytes = kUnitBytes * 2;
	// metadata_pool_bytes deliberately small (not gpu::kDefaultMetadataPoolBytes,
	// 64 MiB) -- Pooled's metadata arena is sized in the same kUnitBytes
	// units as the data arena above, and a tiny kUnitBytes against the
	// *default* 64 MiB metadata budget would round up to hundreds of
	// thousands of units, which was observed to make DeviceContext's own
	// construction hang for minutes building that arena's free-list.
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, /*data_bytes=*/kUnitBytes,
			/*metadata_pool_bytes=*/kUnitBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kUnitBytes);
	gpu::DeviceRegionPool pool(device);
	auto policy = std::make_unique<RecordingAlwaysRejectPolicy>();
	RecordingAlwaysRejectPolicy* observed = policy.get();
	RegionManager manager(std::move(policy));
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	constexpr int kCandidateCount = 5;
	std::vector<std::vector<std::byte>> data(kCandidateCount, std::vector<std::byte>(kRegionBytes, std::byte{0}));
	for (int i = 0; i < kCandidateCount; ++i) {
		HostRegionView host{data[i].data(), kRegionBytes, 0};
		adapter.addRegion(i + 1, host);
		manager.registerRegion(i + 1, host);
		manager.requestPromotion(100 + i, RegionFootprint{{static_cast<RegionId>(i + 1)}});
	}
	// All 5 submitted before any waitIdle() -- processed together in one
	// buildRelocationPlan_collect pass (the policy always Rejects, so the
	// collect loop keeps pulling until the queue is empty, never breaking
	// early).
	manager.waitIdle();

	// Snapshot under lock, then release it -- shutdown() below joins the
	// Coordinator thread, which must remain free to acquire this same
	// mutex_ (e.g. one last hasPendingCandidates() check) on its way out;
	// holding the lock across shutdown() would deadlock the join against
	// that.
	std::vector<const std::vector<EvictionCandidate>*> pointers;
	{
		std::lock_guard<std::mutex> lock(observed->mutex_);
		pointers = observed->observed_pointers;
	}
	ASSERT_EQ(pointers.size(), static_cast<std::size_t>(kCandidateCount));
	for (const auto* ptr : pointers) {
		ASSERT_NE(ptr, nullptr) << "every candidate here needed eviction info -- none should see a null snapshot";
	}
	EXPECT_TRUE(std::all_of(pointers.begin(), pointers.end(),
			[first = pointers.front()](const auto* ptr) { return ptr == first; }))
			<< "every candidate examined within one pass must see the exact same eviction_candidates snapshot "
				 "(shared, not deep-copied) -- see AdmissionContext::eviction_candidates' own doc comment";

	manager.shutdown();
}

// A custom policy exercising ReplacementPolicy::HasExceededPlanningAttempts()
// (the opt-in aging/give-up helper added to the base class this session) --
// proves it's reachable from a subclass and behaves correctly against real
// PromotionCandidate::planning_attempts values RegionManager increments on
// every examination (buildRelocationPlan(), region_manager.cpp), not a
// hand-rolled counter. Always Defers until max_attempts is exceeded, then
// Rejects -- deliberately never actually admits anything, since the point is
// isolating the give-up threshold itself, not promotion.
class GiveUpAfterAttemptsPolicy : public ReplacementPolicy {
 public:
	explicit GiveUpAfterAttemptsPolicy(std::uint64_t max_attempts) : max_attempts_(max_attempts) {}

	void enqueueCandidate(PromotionCandidate candidate) override {
		std::lock_guard<std::mutex> lock(mutex_);
		pending_.push_back(std::move(candidate));
	}
	void requeueCandidate(PromotionCandidate candidate) override {
		std::lock_guard<std::mutex> lock(mutex_);
		pending_.push_back(std::move(candidate));
	}
	void onAnchorEvicted(VectorId anchor_id) override {
		std::lock_guard<std::mutex> lock(mutex_);
		pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
														[anchor_id](const PromotionCandidate& c) { return c.anchor_id == anchor_id; }),
				pending_.end());
	}
	void onAnchorTouched(VectorId) override {}
	bool onRelocationTrigger() override {
		std::lock_guard<std::mutex> lock(mutex_);
		return !pending_.empty();
	}
	bool hasPendingCandidates() const override {
		std::lock_guard<std::mutex> lock(mutex_);
		return !pending_.empty();
	}
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override {
		std::lock_guard<std::mutex> lock(mutex_);
		if (pending_.empty()) return std::nullopt;
		PromotionCandidate candidate = std::move(pending_.front());
		pending_.pop_front();
		return candidate;
	}
	std::optional<VectorId> selectEvictionCandidate(VectorId, std::size_t, const std::vector<EvictionCandidate>&,
			const std::unordered_set<VectorId>& = {}) override {
		return std::nullopt;
	}

	BatchAdmissionDecision evaluateBatchAdmission(const PromotionCandidate& candidate, const AdmissionContext&,
			const RelocationBatchContext&) override {
		std::lock_guard<std::mutex> lock(mutex_);
		attempts_seen.push_back(candidate.planning_attempts);
		if (HasExceededPlanningAttempts(candidate, max_attempts_)) {
			gave_up.push_back(candidate.anchor_id);
			return BatchAdmissionDecision::Reject;
		}
		return BatchAdmissionDecision::Defer;
	}

	mutable std::mutex mutex_;
	std::vector<std::uint64_t> attempts_seen;
	std::vector<VectorId> gave_up;

 private:
	std::uint64_t max_attempts_;
	std::deque<PromotionCandidate> pending_;
};

TEST(RegionManagerCoordinatorTest, HasExceededPlanningAttemptsGivesUpAfterConfiguredThreshold) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	gpu::DeviceContext device;
	gpu::DeviceRegionPool pool(device);
	auto policy = std::make_unique<GiveUpAfterAttemptsPolicy>(/*max_attempts=*/2);
	GiveUpAfterAttemptsPolicy* observed = policy.get();
	RegionManager manager(std::move(policy));
	// Non-forced periodic ticks (not waitIdle()) -- Defer's requeue is only
	// honored on a non-forced/non-stop pass (retain_failed_candidates), so
	// this needs several genuinely separate Coordinator wakeups, one
	// planning_attempts increment apiece, not one forced drain.
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	std::vector<std::byte> host_data(64, std::byte{0x7});
	HostRegionView host{host_data.data(), 64, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(500, RegionFootprint{{1}});
	// Margin for at least 3 separate ticks (2 Defers + 1 Reject).
	std::this_thread::sleep_for(kShortInterval * 40);

	{
		std::lock_guard<std::mutex> lock(observed->mutex_);
		ASSERT_GE(observed->attempts_seen.size(), 3u)
				<< "expected at least 3 separate examinations (2 deferred, 1 rejected)";
		EXPECT_EQ(observed->attempts_seen[0], 1u);
		EXPECT_EQ(observed->attempts_seen[1], 2u);
		EXPECT_EQ(observed->attempts_seen[2], 3u);  // 3 > max_attempts(2) -- this is the one that gives up
		EXPECT_EQ(observed->gave_up, std::vector<VectorId>{500});
	}
	EXPECT_TRUE(manager.regionsOf(500).empty());
	EXPECT_EQ(manager.stats().candidates_rejected_total, 1u);
	// The policy never offers 500 again after rejecting it (see
	// selectNextPromotionCandidate() above -- it's simply gone from
	// pending_), so no further examinations should accumulate even with
	// plenty of time left.
	std::this_thread::sleep_for(kShortInterval * 10);
	{
		std::lock_guard<std::mutex> lock(observed->mutex_);
		EXPECT_EQ(observed->attempts_seen.size(), 3u);
	}

	manager.shutdown();
}

// CoordinatorConfig/RegionManager::Stats::candidates_rejected_total (added
// this session) must count every BatchAdmissionDecision::Reject regardless
// of which policy produced it -- proven here with CostAwareReplacementPolicy
// (the actual default policy), whose own minimum_observations admission gate
// is a real, pre-existing source of Reject decisions, not a test-only stub.
TEST(RegionManagerCoordinatorTest, RejectedCandidateIsCountedAndNeverPromoted) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	gpu::DeviceContext device;
	gpu::DeviceRegionPool pool(device);
	CostAwareReplacementConfig config;
	config.minimum_observations = 5;  // a single request (observations=1) never qualifies
	RegionManager manager(std::make_unique<CostAwareReplacementPolicy>(config));
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	std::vector<std::byte> host_data(64, std::byte{0x7});
	HostRegionView host{host_data.data(), 64, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(600, RegionFootprint{{1}});
	manager.waitIdle();

	EXPECT_FALSE(manager.regionOf(1).device.valid());
	EXPECT_TRUE(manager.regionsOf(600).empty());
	EXPECT_EQ(manager.stats().candidates_rejected_total, 1u);

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, StrictBatchEvictsEnoughVictimsBeforePromotingWholeFootprint) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 2 * kBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	std::vector<std::vector<std::byte>> buffers(4, std::vector<std::byte>(kBytes, std::byte{0}));
	for (RegionId id = 1; id <= 4; ++id) {
		HostRegionView host{buffers[id - 1].data(), kBytes, 0};
		adapter.addRegion(id, host);
		manager.registerRegion(id, host);
	}
	manager.requestPromotion(101, RegionFootprint{{1}});
	manager.waitIdle();
	manager.requestPromotion(102, RegionFootprint{{2}});
	manager.waitIdle();
	manager.requestPromotion(103, RegionFootprint{{3, 4}});
	manager.waitIdle();

	EXPECT_FALSE(manager.regionOf(1).device.valid());
	EXPECT_FALSE(manager.regionOf(2).device.valid());
	EXPECT_TRUE(manager.regionOf(3).device.valid());
	EXPECT_TRUE(manager.regionOf(4).device.valid());
	EXPECT_EQ(manager.stats().anchor_evictions_total, 2u);
	EXPECT_EQ(manager.stats().near_fit_reuses_total, 2u);
	EXPECT_LE(manager.stats().gpu_bytes_allocated, 2 * kBytes);
	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, MoreThanHalfGpuMemoryIsReplacedAsOneLargeNearFitBatch) {
	std::size_t free_bytes = 0;
	std::size_t total_bytes = 0;
	ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);

	constexpr std::size_t kUnits = 4;
	constexpr std::size_t kAlignment = 4096;
	constexpr std::size_t kSafetyMargin = std::size_t{512} << 20;
	std::size_t target_budget = (total_bytes / 100) * 51;
	std::size_t unit_bytes = (target_budget / kUnits / kAlignment) * kAlignment;
	std::size_t budget = unit_bytes * kUnits;
	ASSERT_GT(budget, total_bytes / 2);
	if (free_bytes <= budget + gpu::kDefaultMetadataPoolBytes + kSafetyMargin) {
		GTEST_SKIP() << "requires more than 51% of total GPU memory to be free";
	}

	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, budget,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/4, /*unit_bytes=*/unit_bytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	constexpr std::size_t kLogicalRegionBytes = 4096;
	std::vector<std::vector<std::byte>> buffers(
			8, std::vector<std::byte>(kLogicalRegionBytes, std::byte{0x5a}));
	for (RegionId id = 1; id <= 8; ++id) {
		HostRegionView host{buffers[id - 1].data(), kLogicalRegionBytes, 0};
		adapter.addRegion(id, host);
		manager.registerRegion(id, host);
	}

	std::unordered_set<std::uint64_t> original_handles;
	for (RegionId id = 1; id <= 4; ++id) {
		manager.requestPromotion(100 + id, RegionFootprint{{id}});
		manager.waitIdle();
		Region resident = manager.regionOf(id);
		ASSERT_TRUE(resident.device.valid());
		original_handles.insert(resident.device.id);
	}
	ASSERT_EQ(original_handles.size(), kUnits);
	EXPECT_EQ(pool.bytesReserved(gpu::MemoryKind::Data), budget);
	EXPECT_GT(pool.bytesReserved(gpu::MemoryKind::Data), total_bytes / 2);

	manager.requestPromotion(200, RegionFootprint{{5, 6, 7, 8}});
	manager.waitIdle();

	for (RegionId id = 1; id <= 4; ++id) EXPECT_FALSE(manager.regionOf(id).device.valid());
	for (RegionId id = 5; id <= 8; ++id) {
		Region resident = manager.regionOf(id);
		ASSERT_TRUE(resident.device.valid());
		EXPECT_TRUE(original_handles.contains(resident.device.id));
	}
	EXPECT_EQ(manager.stats().anchor_evictions_total, 4u);
	EXPECT_EQ(manager.stats().near_fit_reuses_total, 4u);
	EXPECT_EQ(pool.bytesReserved(gpu::MemoryKind::Data), budget);
	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, DefaultNearFitThresholdRejectsSlotBelowNinetyPercentUtilization) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kUnitBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 2 * kUnitBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kUnitBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kShortInterval});

	std::vector<std::byte> large_buffer(2 * kUnitBytes, std::byte{0x1});
	std::vector<std::byte> small_buffer(kUnitBytes, std::byte{0x2});
	adapter.addRegion(1, HostRegionView{large_buffer.data(), large_buffer.size(), 0});
	adapter.addRegion(2, HostRegionView{small_buffer.data(), small_buffer.size(), 0});
	manager.registerRegion(1, HostRegionView{large_buffer.data(), large_buffer.size(), 0});
	manager.registerRegion(2, HostRegionView{small_buffer.data(), small_buffer.size(), 0});

	manager.requestPromotion(101, RegionFootprint{{1}});
	manager.waitIdle();
	const std::uint64_t oversized_handle = manager.regionOf(1).device.id;
	manager.requestPromotion(102, RegionFootprint{{2}});
	manager.waitIdle();

	ASSERT_TRUE(manager.regionOf(2).device.valid());
	EXPECT_NE(manager.regionOf(2).device.id, oversized_handle);
	EXPECT_EQ(manager.stats().near_fit_reuses_total, 0u);
	EXPECT_EQ(pool.bytesReserved(gpu::MemoryKind::Data), kUnitBytes);
	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, NearFitThresholdCanBeConfiguredAtInitialization) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kUnitBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, 2 * kUnitBytes,
			gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, /*unit_bytes=*/kUnitBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager(std::make_unique<FifoReplacementPolicy>());
	CoordinatorConfig config{kShortInterval};
	config.near_fit_min_utilization_percent = 50;
	manager.start(adapter, pool, routing_cache, config);

	std::vector<std::byte> large_buffer(2 * kUnitBytes, std::byte{0x1});
	std::vector<std::byte> small_buffer(kUnitBytes, std::byte{0x2});
	adapter.addRegion(1, HostRegionView{large_buffer.data(), large_buffer.size(), 0});
	adapter.addRegion(2, HostRegionView{small_buffer.data(), small_buffer.size(), 0});
	manager.registerRegion(1, HostRegionView{large_buffer.data(), large_buffer.size(), 0});
	manager.registerRegion(2, HostRegionView{small_buffer.data(), small_buffer.size(), 0});

	manager.requestPromotion(101, RegionFootprint{{1}});
	manager.waitIdle();
	const std::uint64_t reusable_handle = manager.regionOf(1).device.id;
	manager.requestPromotion(102, RegionFootprint{{2}});
	manager.waitIdle();

	ASSERT_TRUE(manager.regionOf(2).device.valid());
	EXPECT_EQ(manager.regionOf(2).device.id, reusable_handle);
	EXPECT_EQ(manager.stats().near_fit_reuses_total, 1u);
	EXPECT_EQ(pool.bytesReserved(gpu::MemoryKind::Data), 2 * kUnitBytes);
	manager.shutdown();
}

// ---------------------------------------------------------------------------
// PromotionCandidate's owned vector copy + per-Anchor epoch (see its own doc
// comment, replacement_policy.hpp): the two mechanisms the RoutingCache-
// registration move into RegionManager required.
// ---------------------------------------------------------------------------

TEST(RegionManagerCoordinatorTest, PromotionRegistersAnchorInRoutingCacheWithAnOwnedVectorCopyOutlivingTheCaller) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	{
		// This buffer goes out of scope right after requestPromotion()
		// returns, like a real caller's stack-local Query/Record (VectorView
		// is non-owning, see types.hpp). requestPromotion() must copy the
		// bytes, not just borrow the pointer, or the deferred grant below would read freed memory.
		std::vector<float> vec{1.0f, 2.0f, 3.0f};
		manager.requestPromotion(100, RegionFootprint{{1}}, VectorView{vec.data(), 3, VectorDType::Float32});
	}

	manager.waitIdle();
	ASSERT_EQ(routing_cache.ensured, std::vector<VectorId>{100});
	ASSERT_EQ(routing_cache.last_ensure_bytes.size(), 3 * sizeof(float));
	std::vector<float> recovered(3);
	std::memcpy(recovered.data(), routing_cache.last_ensure_bytes.data(), recovered.size() * sizeof(float));
	EXPECT_EQ(recovered, (std::vector<float>{1.0f, 2.0f, 3.0f}));

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, StaleEpochCandidateIsDiscardedAfterReleaseAnchorRacesAheadOfTheCoordinator) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	// Simulates Controller::insert()'s race: a promotion candidate gets
	// enqueued (e.g. dispatch()'s on_complete callback), then the same
	// anchor is released (e.g. remove()) before the Coordinator ever
	// drains pending_promotions_ into the policy.
	manager.requestPromotion(100, RegionFootprint{{1}});
	manager.releaseAnchor(100);  // bumps anchor 100's epoch -- the stamped candidate is now stale

	manager.waitIdle();

	// The stale candidate must never have been granted: no dependency, no
	// GPU residency, no RoutingCache registration for it.
	EXPECT_TRUE(manager.regionsOf(100).empty());
	EXPECT_FALSE(manager.regionOf(1).device.valid());
	EXPECT_EQ(manager.stats().regions_promoted_total, 0u);
	EXPECT_TRUE(routing_cache.ensured.empty());

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, ReAdmittingAnAnchorAfterReleaseIsNotTreatedAsStale) {
	// The epoch bump must not permanently blacklist a VectorId -- a brand-new
	// insert() legitimately reusing a freed id must promote normally.
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes,
														 /*worker_stream_count=*/1, /*unit_bytes=*/kBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	manager.requestPromotion(100, RegionFootprint{{1}});
	manager.releaseAnchor(100);
	manager.requestPromotion(100, RegionFootprint{{1}});  // re-admitted with the current epoch

	manager.waitIdle();

	EXPECT_EQ(manager.regionsOf(100).size(), 1u);
	EXPECT_TRUE(manager.regionOf(1).device.valid());
	EXPECT_EQ(routing_cache.ensured, std::vector<VectorId>{100});

	manager.shutdown();
}

}  // namespace
