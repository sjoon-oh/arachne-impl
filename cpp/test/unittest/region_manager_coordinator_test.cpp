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
	std::optional<VectorId> selectNextEvictionCandidate(VectorId) override { return std::nullopt; }
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
