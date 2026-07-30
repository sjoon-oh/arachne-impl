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

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
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

// Records every ensure()/erase() call verbatim -- lets tests assert exactly
// when RegionManager registers/removes an Anchor in RoutingCache (see
// region_manager.hpp's class doc comment: this now happens at actual
// promotion-grant/eviction time, not unconditionally on every query the way
// Controller's old commitSearch()/commitInsert() did it). "No match" for
// nearest() is enough here -- these tests never exercise Controller's own
// routing decision.
class FakeRoutingCache : public RoutingCache {
 public:
	FakeRoutingCache() : RoutingCache(/*dim=*/1, DistanceMetric::L2, VectorDType::Float32) {}

	std::optional<VectorId> nearest(const VectorView&) override { return std::nullopt; }
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
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes);
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

TEST(RegionManagerCoordinatorTest, ReleaseAnchorBookkeepingIsImmediateButReclaimIsLazy) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes);
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
	// Dependency-graph bookkeeping and residency clearing are synchronous --
	// see releaseAnchor()'s own doc comment for why this matters (so a
	// concurrent requestPromotion() for a different Anchor onto the same
	// Region re-promotes fresh rather than seeing a stale "already
	// promoted" lease). RoutingCache erasure is synchronous too.
	EXPECT_TRUE(manager.regionsOf(100).empty());
	EXPECT_FALSE(manager.regionOf(1).device.valid());
	EXPECT_EQ(routing_cache.erased, std::vector<VectorId>{100});

	// The actual GPU free() is deferred -- force it and check it landed.
	manager.waitIdle();
	EXPECT_EQ(manager.stats().anchor_evictions_total, 1u);
	EXPECT_EQ(pool.bytesAllocated(), 0u);

	manager.shutdown();
}

TEST(RegionManagerCoordinatorTest, TimelyTriggerProcessesWithoutExplicitWaitIdle) {
	// "Timely event triggering": the Coordinator's own periodic tick, with no
	// caller ever calling waitIdle(), must eventually catch up on its own.
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes);
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
														 gpu::kDefaultMetadataPoolBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
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

	manager.requestPromotion(103, RegionFootprint{{3}});
	manager.waitIdle();  // needs capacity -- evicts anchor 101 (FIFO-oldest)

	EXPECT_FALSE(manager.regionOf(1).device.valid());  // evicted
	EXPECT_TRUE(manager.regionOf(2).device.valid());   // untouched
	EXPECT_TRUE(manager.regionOf(3).device.valid());   // newly promoted
	EXPECT_EQ(manager.stats().anchor_evictions_total, 1u);
	EXPECT_LE(manager.stats().gpu_bytes_allocated, 2 * kBytes);
	// Capacity-driven eviction (evictAnchorNow(), not releaseAnchor()) must
	// also erase from RoutingCache -- anchor 101 no longer represents live
	// GPU residency.
	EXPECT_EQ(routing_cache.erased, std::vector<VectorId>{101});
	EXPECT_EQ(routing_cache.ensured, (std::vector<VectorId>{101, 102, 103}));

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
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	{
		// This buffer goes out of scope immediately after requestPromotion()
		// returns -- exactly like a real caller's stack-local Query/Record
		// vector would (VectorView is explicitly non-owning, see types.hpp).
		// requestPromotion() must copy the bytes out, not just borrow the
		// pointer, or this Coordinator-thread grant (kLongInterval, forced by
		// waitIdle() below, well after this scope ends) would read freed
		// memory.
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
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes);
	gpu::DeviceRegionPool pool(device);
	RegionManager manager;
	manager.start(adapter, pool, routing_cache, CoordinatorConfig{kLongInterval});

	std::vector<std::byte> host_data(kBytes, std::byte{0x7});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	manager.registerRegion(1, host);

	// Simulates Controller::insert()'s own race: a promotion candidate is
	// enqueued (e.g. by dispatch()'s on_complete callback right after the
	// lookup traversal), but the same anchor is released (e.g. remove())
	// before the Coordinator (kLongInterval, no waitIdle() yet) ever drains
	// pending_promotions_ into the policy.
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
	gpu::DeviceContext device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, kBytes, gpu::kDefaultMetadataPoolBytes);
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
