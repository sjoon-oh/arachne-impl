// Deterministic concurrency stress for RegionManager's residency protocol.
// Unlike broad random churn tests, these tests place barriers at exact
// routing/validation/execution boundaries so a failed interleaving is
// reproducible rather than timing-dependent.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "core/op_scheduler.hpp"
#include "core/region_manager.hpp"
#include "core/routing_cache.hpp"
#include "gpu/device_context.hpp"
#include "gpu/device_region_pool.hpp"

namespace {

using namespace arachne;
using namespace std::chrono_literals;

class ArrivalGate {
 public:
	void arriveAndWait() {
		std::unique_lock<std::mutex> lock(mutex_);
		++arrivals_;
		cv_.notify_all();
		cv_.wait(lock, [this] { return open_; });
	}

	bool waitForArrivals(std::size_t expected, std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> lock(mutex_);
		return cv_.wait_for(lock, timeout, [this, expected] { return arrivals_ >= expected; });
	}

	void open() {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			open_ = true;
		}
		cv_.notify_all();
	}

 private:
	std::mutex mutex_;
	std::condition_variable cv_;
	std::size_t arrivals_ = 0;
	bool open_ = false;
};

class RaceRegion final : public IRegion {
 public:
	RaceRegion() = default;
	RaceRegion(RegionId id, HostRegionView host) : id_(id), host_(host) {}

	RegionId id() const override { return id_; }
	RegionFootprint footprint() const override { return RegionFootprint{{id_}}; }
	HostRegionView hostView() const override { return host_; }
	LeaseHandle acquireWriteLease() override { return LeaseHandle{id_, ++epoch_}; }
	void releaseWriteLease(LeaseHandle) override {}
	void applyLocalModification(LeaseHandle, const ModificationDelta&) override {}
	ReconciliationReport reconcileBoundary() override { return {}; }

 private:
	RegionId id_ = 0;
	HostRegionView host_;
	std::uint64_t epoch_ = 0;
};

class RaceAdapter final : public IAdapter {
 public:
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		host_calls_.fetch_add(requests.size(), std::memory_order_relaxed);
		std::vector<TraverseResult> results(requests.size());
		for (TraverseResult& result : results) result.completed_within_scope = true;
		return results;
	}

	std::vector<TraverseResult> traverseDevice(const std::vector<TraverseRequest>& requests) override {
		device_calls_.fetch_add(requests.size(), std::memory_order_relaxed);
		if (device_gate_ != nullptr) device_gate_->arriveAndWait();
		std::vector<TraverseResult> results(requests.size());
		for (TraverseResult& result : results) result.completed_within_scope = true;
		return results;
	}

	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		return std::vector<ModifyResult>(requests.size());
	}

	IRegion* resolveRegion(RegionId id) override {
		auto it = regions_.find(id);
		return it == regions_.end() ? nullptr : &it->second;
	}

	std::vector<RegionId> allRegions() const override {
		std::vector<RegionId> result;
		result.reserve(regions_.size());
		for (const auto& [id, region] : regions_) result.push_back(id);
		return result;
	}

	void addRegion(RegionId id, HostRegionView host) { regions_.emplace(id, RaceRegion(id, host)); }
	void setDeviceGate(ArrivalGate* gate) { device_gate_ = gate; }
	std::size_t hostCalls() const { return host_calls_.load(std::memory_order_relaxed); }
	std::size_t deviceCalls() const { return device_calls_.load(std::memory_order_relaxed); }

 private:
	std::unordered_map<RegionId, RaceRegion> regions_;
	ArrivalGate* device_gate_ = nullptr;
	std::atomic<std::size_t> host_calls_{0};
	std::atomic<std::size_t> device_calls_{0};
};

class RaceRoutingCache final : public RoutingCache {
 public:
	RaceRoutingCache() : RoutingCache(/*dim=*/1, DistanceMetric::L2, VectorDType::Float32) {}

	VectorId ensure(VectorId id, const VectorView&, float) override { return id; }
	void erase(VectorId) override {}

 protected:
	std::optional<VectorId> nearestImpl(const VectorView&) override { return std::nullopt; }
};

struct RaceHarness {
	static constexpr std::size_t kRegionBytes = 4096;

	explicit RaceHarness(std::size_t region_count = 1)
			: host_buffers(region_count, std::vector<std::byte>(kRegionBytes, std::byte{0x5a})),
				device(/*device_id=*/0, gpu::AllocationPolicy::Pooled, region_count * kRegionBytes,
						 gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/4,
						 /*unit_bytes=*/kRegionBytes),
				pool(device) {
		manager.start(adapter, pool, routing_cache, CoordinatorConfig{1ms});
		for (std::size_t i = 0; i < region_count; ++i) {
			RegionId id = static_cast<RegionId>(i + 1);
			HostRegionView host{host_buffers[i].data(), kRegionBytes, 0};
			adapter.addRegion(id, host);
			manager.registerRegion(id, host);
		}
	}

	~RaceHarness() { manager.shutdown(); }

	void promote(VectorId anchor, RegionId region) {
		manager.requestPromotion(anchor, RegionFootprint{{region}});
		manager.waitIdle();
	}

	std::vector<std::vector<std::byte>> host_buffers;
	RaceAdapter adapter;
	RaceRoutingCache routing_cache;
	gpu::DeviceContext device;
	gpu::DeviceRegionPool pool;
	RegionManager manager;
};

void ValidateAndPin(RegionManager& manager, TraverseRequest& request) {
	if (request.mode != ExecutionMode::GpuOnly) return;
	request.residency_pin = manager.tryPinResidency(request.residency_hints);
	if (!request.residency_pin) {
		request.mode = ExecutionMode::Hybrid;
		request.scope = {};
	}
}

TraverseRequest DeviceRequest(const std::vector<RegionResidencyHint>& hints) {
	TraverseRequest request;
	request.mode = ExecutionMode::GpuOnly;
	request.residency_hints = hints;
	for (const RegionResidencyHint& hint : hints) request.scope.regions.push_back(hint.region);
	return request;
}

TEST(RegionResidencyConcurrencyStressTest, MultipleWorkersPinningOneVictimDrainBeforeReclaim) {
	constexpr std::size_t kWorkers = 4;
	constexpr VectorId kAnchor = 101;
	RaceHarness harness;
	harness.promote(kAnchor, /*region=*/1);
	std::vector<RegionResidencyHint> hints = harness.manager.residencyHints(kAnchor);
	ASSERT_EQ(hints.size(), 1u);

	ArrivalGate device_gate;
	harness.adapter.setDeviceGate(&device_gate);
	SchedulingConfig config;
	config.traverse_batch_size = 1;
	config.max_execution_threads = kWorkers;
	OpScheduler scheduler(config);
	scheduler.start(harness.adapter, nullptr,
			[&harness](TraverseRequest& request) { ValidateAndPin(harness.manager, request); });

	std::vector<std::future<TraverseResult>> futures;
	for (std::size_t i = 0; i < kWorkers; ++i) futures.push_back(scheduler.schedule(DeviceRequest(hints)));

	bool all_workers_pinned = device_gate.waitForArrivals(kWorkers, 5s);
	if (!all_workers_pinned) {
		device_gate.open();
		scheduler.shutdown();
		FAIL() << "not every worker reached the device adapter";
	}

	harness.manager.releaseAnchor(kAnchor);
	Region retiring = harness.manager.regionOf(1);
	EXPECT_EQ(retiring.residency_state, RegionResidencyState::Retiring);
	EXPECT_EQ(retiring.residency_pins, kWorkers);
	EXPECT_TRUE(retiring.device.valid());
	EXPECT_EQ(harness.pool.bytesAllocated(), RaceHarness::kRegionBytes);

	device_gate.open();
	for (std::future<TraverseResult>& future : futures) {
		EXPECT_EQ(future.get().execution_mode, ExecutionMode::GpuOnly);
	}
	scheduler.shutdown();
	harness.manager.waitIdle();

	Region reclaimed = harness.manager.regionOf(1);
	EXPECT_EQ(reclaimed.residency_state, RegionResidencyState::HostOnly);
	EXPECT_EQ(reclaimed.residency_pins, 0u);
	EXPECT_FALSE(reclaimed.device.valid());
	EXPECT_EQ(harness.pool.bytesAllocated(), 0u);
	EXPECT_EQ(harness.adapter.deviceCalls(), kWorkers);
}

TEST(RegionResidencyConcurrencyStressTest, EvictionBetweenRoutingAndValidationDeterministicallyFallsBack) {
	constexpr VectorId kAnchor = 202;
	RaceHarness harness;
	harness.promote(kAnchor, /*region=*/1);
	std::vector<RegionResidencyHint> routed_hints = harness.manager.residencyHints(kAnchor);
	ASSERT_EQ(routed_hints.size(), 1u);
	std::uint64_t routed_generation = routed_hints.front().generation;

	ArrivalGate before_validation;
	SchedulingConfig config;
	config.max_execution_threads = 1;
	OpScheduler scheduler(config);
	scheduler.start(harness.adapter, nullptr, [&harness, &before_validation](TraverseRequest& request) {
		before_validation.arriveAndWait();
		ValidateAndPin(harness.manager, request);
	});

	std::future<TraverseResult> future = scheduler.schedule(DeviceRequest(routed_hints));
	bool reached_validation_boundary = before_validation.waitForArrivals(1, 5s);
	if (!reached_validation_boundary) {
		before_validation.open();
		scheduler.shutdown();
		FAIL() << "worker did not reach the pre-validation boundary";
	}

	// The request already carries a routing snapshot, but has not validated
	// or pinned it yet. Complete eviction inside this exact gap.
	harness.manager.releaseAnchor(kAnchor);
	harness.manager.waitIdle();
	Region evicted = harness.manager.regionOf(1);
	ASSERT_EQ(evicted.residency_state, RegionResidencyState::HostOnly);
	ASSERT_GT(evicted.residency_generation, routed_generation);

	before_validation.open();
	TraverseResult result = future.get();
	scheduler.shutdown();

	EXPECT_EQ(result.execution_mode, ExecutionMode::Hybrid);
	EXPECT_EQ(harness.adapter.hostCalls(), 1u);
	EXPECT_EQ(harness.adapter.deviceCalls(), 0u);
	EXPECT_EQ(harness.pool.bytesAllocated(), 0u);
}

TEST(RegionResidencyConcurrencyStressTest, SustainedRepinningCannotStarveEviction) {
	constexpr VectorId kAnchor = 303;
	constexpr std::size_t kPinners = 8;
	constexpr std::uint64_t kWarmupPins = 10000;
	RaceHarness harness;
	harness.promote(kAnchor, /*region=*/1);
	std::vector<RegionResidencyHint> hints = harness.manager.residencyHints(kAnchor);
	ASSERT_EQ(hints.size(), 1u);

	std::atomic<bool> stop{false};
	std::atomic<std::uint64_t> successful_pins{0};
	std::atomic<std::uint64_t> rejected_pins{0};
	std::vector<std::thread> pinners;
	for (std::size_t i = 0; i < kPinners; ++i) {
		pinners.emplace_back([&] {
			while (!stop.load(std::memory_order_relaxed)) {
				std::shared_ptr<void> pin = harness.manager.tryPinResidency(hints);
				if (pin) {
					successful_pins.fetch_add(1, std::memory_order_relaxed);
					std::this_thread::sleep_for(25us);
				} else {
					rejected_pins.fetch_add(1, std::memory_order_relaxed);
					std::this_thread::yield();
				}
			}
		});
	}

	auto warmup_deadline = std::chrono::steady_clock::now() + 5s;
	while (successful_pins.load(std::memory_order_relaxed) < kWarmupPins &&
			 std::chrono::steady_clock::now() < warmup_deadline) {
		std::this_thread::sleep_for(1ms);
	}
	EXPECT_GE(successful_pins.load(std::memory_order_relaxed), kWarmupPins);

	// Pinners keep running while retirement begins. Resident -> Retiring is
	// the fairness boundary: every later tryPinResidency() must fail, allowing
	// only the finite set of already-held pins to drain.
	harness.manager.releaseAnchor(kAnchor);
	auto idle = std::async(std::launch::async, [&harness] { harness.manager.waitIdle(); });
	std::future_status status = idle.wait_for(5s);

	stop.store(true, std::memory_order_relaxed);
	for (std::thread& pinner : pinners) pinner.join();
	if (status == std::future_status::ready) {
		idle.get();
	} else {
		// Avoid leaving a background waiter behind if the assertion fails.
		harness.manager.waitIdle();
		idle.get();
	}

	EXPECT_EQ(status, std::future_status::ready) << "eviction starved under continuous pin pressure";
	EXPECT_GT(rejected_pins.load(std::memory_order_relaxed), 0u);
	Region reclaimed = harness.manager.regionOf(1);
	EXPECT_EQ(reclaimed.residency_state, RegionResidencyState::HostOnly);
	EXPECT_EQ(reclaimed.residency_pins, 0u);
	EXPECT_FALSE(reclaimed.device.valid());
	EXPECT_EQ(harness.pool.bytesAllocated(), 0u);
}

}  // namespace
