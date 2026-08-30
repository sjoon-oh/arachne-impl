#include "core/controller.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "adapter/region.hpp"
#include "core/routing_cache.hpp"
#include "gpu/device_region_pool.hpp"
#include "gpu/dirty_header.hpp"
#include "types.hpp"

namespace {

using namespace arachne;

// -----------------------------------------------------------------------------
// Tests for Controller's GPU-residency machinery: promotion, eviction, and
// dirty-region write-back, driven purely through insert()/acquireRegion()/
// remove() against minimal test doubles (no real index, no real GPU adapter
// logic).
//
// Test doubles:
//  - FakeRegion / FakeAdapter: a minimal IRegion/IAdapter pair. FakeAdapter's
//    traverseHost() reports whatever RegionFootprint the test sets on
//    next_touched just before each insert(), standing in for a real index's
//    locality decision; modifyHost() always succeeds and echoes back the
//    request's scope.
//  - FakeRoutingCache: always reports "no match", keeping every insert on
//    the Hybrid path so tests exercise promoteAnchor()'s capacity/eviction
//    logic via commitInsert(), not query routing.
//  - MakeRecord(): builds a single-float Record for insert().
//  - PokeDevice(): writes bytes directly into a region's device allocation
//    via acquireRegion(), simulating "a kernel just wrote here" (payload) or
//    "a kernel just atomicOr'd a dirty bit" (header), so eviction's
//    writeBackDirtyRegions() dirty-vs-clean behavior can be tested.
//
// ControllerGpuResidencyTest covers: promotion allocates GPU memory and
// copies host data; eviction reclaims exactly as many victims as needed
// (including multi-victim evictions); eviction writes back only dirty
// regions (or always, when no dirty header exists); eviction batches
// multiple regions belonging to one anchor into a single evictAnchor() call;
// and insert() rejects duplicate ids without disturbing existing state.
// -----------------------------------------------------------------------------

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

// traverseHost() reports next_touched as every TraverseResult::touched,
// driving promotion (see Controller::commitInsert()); modifyHost() always
// succeeds and echoes the request's scope back as ModifyResult::touched.
class FakeAdapter : public IAdapter {
 public:
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		std::vector<TraverseResult> results(requests.size());
		for (TraverseResult& result : results) result.touched = next_touched;
		return results;
	}

	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		std::vector<ModifyResult> results;
		results.reserve(requests.size());
		for (const ModifyRequest& request : requests) {
			ModifyResult result;
			result.ok = true;
			result.touched = request.scope;
			results.push_back(result);
		}
		return results;
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

	RegionFootprint next_touched;

 private:
	std::unordered_map<RegionId, FakeRegion> regions_;
};

class FakeRoutingCache : public RoutingCache {
 public:
	FakeRoutingCache() : RoutingCache(/*dim=*/1, DistanceMetric::L2, VectorDType::Float32) {}

	// Always "no match": keeps every query/insert lookup on the Hybrid path,
	// which is all these tests need -- they exercise promoteAnchor()'s
	// capacity/eviction logic via commitInsert(), not query routing itself.
	std::optional<VectorId> nearestImpl(const VectorView&) override { return std::nullopt; }
	VectorId ensure(VectorId id, const VectorView&, float) override { return id; }
	void erase(VectorId) override {}
};

// FakeRoutingCache plus bookkeeping of exactly which ids Controller registered
// (ensure()) or released (erase()) an Anchor under -- lets a test inspect the
// actual anchor_id Controller minted/used, not just the resulting residency.
class RecordingRoutingCache : public FakeRoutingCache {
 public:
	VectorId ensure(VectorId id, const VectorView& vector, float max_distance) override {
		ensured_ids.push_back(id);
		return FakeRoutingCache::ensure(id, vector, max_distance);
	}
	void erase(VectorId id) override {
		erased_ids.push_back(id);
		FakeRoutingCache::erase(id);
	}

	std::vector<VectorId> ensured_ids;
	std::vector<VectorId> erased_ids;
};

// Delegates every call to a real FifoReplacementPolicy, counting
// onAnchorEvicted() calls along the way -- lets a test assert a code path
// did (or, per the fix this file is testing, deliberately does *not*)
// reach the ReplacementPolicy at all. Can't just subclass
// FifoReplacementPolicy and override one method: it's declared `final`.
class EvictionCountingReplacementPolicy : public ReplacementPolicy {
 public:
	void enqueueCandidate(PromotionCandidate candidate) override { inner_.enqueueCandidate(std::move(candidate)); }
	void requeueCandidate(PromotionCandidate candidate) override { inner_.requeueCandidate(std::move(candidate)); }
	void onAnchorEvicted(VectorId anchor_id) override {
		++eviction_calls;
		inner_.onAnchorEvicted(anchor_id);
	}
	void onAnchorTouched(VectorId anchor_id) override { inner_.onAnchorTouched(anchor_id); }
	bool onRelocationTrigger() override { return inner_.onRelocationTrigger(); }
	bool hasPendingCandidates() const override { return inner_.hasPendingCandidates(); }
	std::optional<PromotionCandidate> selectNextPromotionCandidate() override {
		return inner_.selectNextPromotionCandidate();
	}
	std::optional<VectorId> selectEvictionCandidate(VectorId excluded, std::size_t required_bytes,
																									 const std::vector<EvictionCandidate>& candidates) override {
		return inner_.selectEvictionCandidate(excluded, required_bytes, candidates);
	}

	int eviction_calls = 0;

 private:
	FifoReplacementPolicy inner_;
};

// Mirrors controller.cpp's own kAnchorIdBit -- the top bit of VectorId,
// reserved by MintAnchorId() so a minted Anchor id can never numerically
// collide with a real data-vector id (see MintAnchorId()'s and
// next_anchor_id_'s doc comments in controller.hpp). Redeclared here
// deliberately rather than exposed from controller.cpp: it's the public
// contract TraverseRequest::anchor_id's doc comment already promises
// (index_adapter.hpp) -- "not guaranteed to be a real element" -- and these
// tests are checking that promise holds, not reaching into an internal.
constexpr VectorId kAnchorIdTopBit = VectorId{1} << 63;

// Builds a Record whose vector points at `value` -- callers must keep
// `value` alive for the duration of the Controller::insert() call, which is
// synchronous, so a stack float held by the test body is enough.
Record MakeRecord(VectorId id, const float& value) {
	Record record;
	record.id = id;
	record.vector = VectorView{&value, /*dim=*/1, VectorDType::Float32};
	return record;
}

// Writes `bytes` directly into `region`'s device allocation via
// acquireRegion(), simulating "a kernel wrote here" (payload) or "a kernel
// atomicOr'd a dirty bit" (header). The Lease releases before returning so
// a later eviction's free() doesn't block on it.
void PokeDevice(Controller& controller, RegionId region, std::size_t offset, const std::vector<std::byte>& bytes) {
	RegionAccess access = controller.acquireRegion(region);
	ASSERT_TRUE(access.on_device);
	gpu::DeviceRegionPool::Lease& lease = *access.device_lease;
	ASSERT_EQ(cudaMemcpyAsync(static_cast<std::byte*>(lease.ptr()) + offset, bytes.data(), bytes.size(),
														 cudaMemcpyHostToDevice, lease.stream()),
						cudaSuccess);
	ASSERT_EQ(cudaStreamSynchronize(lease.stream()), cudaSuccess);
}

// ---------------------------------------------------------------------------

TEST(ControllerGpuResidencyTest, PromoteAllocatesGpuMemoryAndCopiesHostData) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	Controller controller(adapter, routing_cache);

	constexpr std::size_t kBytes = 64;
	std::vector<std::byte> host_data(kBytes, std::byte{0x42});
	HostRegionView host{host_data.data(), kBytes, /*subregion_bytes=*/0};
	adapter.addRegion(/*region=*/1, host);
	controller.registerRegion(/*id=*/1, host);

	// Before any promotion, the region is host-only.
	RegionAccess before = controller.acquireRegion(1);
	EXPECT_FALSE(before.on_device);

	adapter.next_touched = RegionFootprint{{1}};
	float vector_value = 1.0f;
	InsertResult result = controller.insert(MakeRecord(/*id=*/100, vector_value));
	ASSERT_TRUE(result.ok);
	controller.waitIdle();  // promotion is now lazy -- wait for RegionManager's Coordinator to catch up

	RegionAccess after = controller.acquireRegion(1);
	ASSERT_TRUE(after.on_device);
	ASSERT_TRUE(after.device_lease.has_value());

	std::vector<std::byte> device_out(kBytes);
	ASSERT_EQ(cudaMemcpyAsync(device_out.data(), after.device_lease->ptr(), kBytes, cudaMemcpyDeviceToHost,
														 after.device_lease->stream()),
						cudaSuccess);
	ASSERT_EQ(cudaStreamSynchronize(after.device_lease->stream()), cudaSuccess);
	EXPECT_EQ(device_out, host_data);
}

TEST(ControllerGpuResidencyTest, PromoteEvictsMultipleVictimsWhenOneIsNotEnoughCapacity) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	// Budget holds exactly 3 small (256B) regions; promoting a 4th, big
	// (512B) region can't be satisfied by evicting just one (256 < 512), so
	// this proves promoteAnchor()'s eviction loop keeps going past a single
	// victim. gpu_unit_bytes == kSmallBytes avoids unit-rounding slack.
	constexpr std::size_t kSmallBytes = 256;
	constexpr std::size_t kBigBytes = 512;
	Controller controller(adapter, routing_cache, SchedulingConfig{}, std::make_unique<FifoReplacementPolicy>(),
												 /*gpu_data_budget_bytes=*/3 * kSmallBytes, gpu::kDefaultMetadataPoolBytes,
												 /*gpu_unit_bytes=*/kSmallBytes, /*compaction_policy=*/nullptr, /*coordinator_config=*/{},
												 gpu::AllocationPolicy::Pooled);

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
	controller.registerRegion(1, host_a);
	controller.registerRegion(2, host_b);
	controller.registerRegion(3, host_c);
	controller.registerRegion(4, host_d);

	float v1 = 1.0f, v2 = 2.0f, v3 = 3.0f, v4 = 4.0f;

	adapter.next_touched = RegionFootprint{{1}};
	ASSERT_TRUE(controller.insert(MakeRecord(101, v1)).ok);  // record 101's insert-traverse anchor -> region 1 (oldest)
	adapter.next_touched = RegionFootprint{{2}};
	ASSERT_TRUE(controller.insert(MakeRecord(102, v2)).ok);  // record 102's insert-traverse anchor -> region 2
	adapter.next_touched = RegionFootprint{{3}};
	ASSERT_TRUE(controller.insert(MakeRecord(103, v3)).ok);  // record 103's insert-traverse anchor -> region 3 -- budget now full

	adapter.next_touched = RegionFootprint{{4}};
	ASSERT_TRUE(controller.insert(MakeRecord(104, v4)).ok);  // record 104's insert-traverse anchor -> region 4, needs 2 evictions
	controller.waitIdle();  // promotion/eviction is now lazy -- wait for the Coordinator to catch up

	EXPECT_FALSE(controller.acquireRegion(1).on_device);  // evicted (oldest, FIFO)
	EXPECT_FALSE(controller.acquireRegion(2).on_device);  // evicted (second-oldest)
	EXPECT_TRUE(controller.acquireRegion(3).on_device);   // untouched -- one eviction wasn't enough, but two was
	EXPECT_TRUE(controller.acquireRegion(4).on_device);   // newly promoted
}

// End-to-end proof that CoordinatorConfig::max_promotion_fraction_of_budget
// (region_manager.hpp) is correctly resolved by Controller's constructor --
// not just stored, but actually converted into the real byte cap
// RegionManager's Coordinator enforces. Four candidates that would otherwise
// all fit in the budget with room to spare (so this isn't testing eviction
// at all, unlike the test above) are forced across multiple relocation
// passes within one waitIdle() call purely by the resolved promotion cap.
TEST(ControllerGpuResidencyTest, MaxPromotionFractionOfBudgetResolvesAndSplitsAPass) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kSmallBytes = 256;

	CoordinatorConfig config;
	config.trigger_interval = std::chrono::milliseconds(200);
	// budget = 4*256 = 1024B; 0.26 resolves to ~266B -- barely over one
	// region, so a second same-sized candidate can never join the same pass.
	config.max_promotion_fraction_of_budget = 0.26;

	Controller controller(adapter, routing_cache, SchedulingConfig{}, std::make_unique<FifoReplacementPolicy>(),
			/*gpu_data_budget_bytes=*/4 * kSmallBytes, gpu::kDefaultMetadataPoolBytes,
			/*gpu_unit_bytes=*/kSmallBytes, /*compaction_policy=*/nullptr, config, gpu::AllocationPolicy::Pooled);

	std::vector<std::vector<std::byte>> buffers(4, std::vector<std::byte>(kSmallBytes, std::byte{0}));
	for (RegionId id = 1; id <= 4; ++id) {
		HostRegionView host{buffers[id - 1].data(), kSmallBytes, 0};
		adapter.addRegion(id, host);
		controller.registerRegion(id, host);
	}

	float v1 = 1.0f, v2 = 2.0f, v3 = 3.0f, v4 = 4.0f;
	adapter.next_touched = RegionFootprint{{1}};
	ASSERT_TRUE(controller.insert(MakeRecord(401, v1)).ok);
	adapter.next_touched = RegionFootprint{{2}};
	ASSERT_TRUE(controller.insert(MakeRecord(402, v2)).ok);
	adapter.next_touched = RegionFootprint{{3}};
	ASSERT_TRUE(controller.insert(MakeRecord(403, v3)).ok);
	adapter.next_touched = RegionFootprint{{4}};
	ASSERT_TRUE(controller.insert(MakeRecord(404, v4)).ok);
	// No intermediate waitIdle() -- all four requests reach the Coordinator's
	// pending set before it wakes up, same as the multi-victim test above.
	controller.waitIdle();

	EXPECT_TRUE(controller.acquireRegion(1).on_device);
	EXPECT_TRUE(controller.acquireRegion(2).on_device);
	EXPECT_TRUE(controller.acquireRegion(3).on_device);
	EXPECT_TRUE(controller.acquireRegion(4).on_device);
	EXPECT_GT(controller.stats().relocation_batches_total, 1u)
			<< "the resolved ~266B cap should have forced more than one Coordinator pass for 4*256B of candidates";
	EXPECT_GT(controller.stats().candidates_requeued_total, 0u)
			<< "at least one candidate should have been bumped by the cap and requeued rather than lost";
}

TEST(ControllerGpuResidencyTest, EvictionWritesBackOnlyWhenDirty) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	// Two regions with subregion tracking enabled (a real dirty header
	// exists): 248B payload + 124B subregions -> one 8B header word -> 256B
	// per region. Budget holds exactly 2 (gpu_unit_bytes == region_bytes).
	constexpr std::size_t kPayloadBytes = 248;
	constexpr std::size_t kSubregionBytes = 124;
	const std::size_t header_bytes = gpu::DirtyHeaderBytes(kPayloadBytes, kSubregionBytes);
	ASSERT_EQ(header_bytes, gpu::kDirtyWordBytes);
	const std::size_t region_bytes = header_bytes + kPayloadBytes;
	ASSERT_EQ(region_bytes, 256u);

	Controller controller(adapter, routing_cache, SchedulingConfig{}, nullptr,
												 /*gpu_data_budget_bytes=*/2 * region_bytes, gpu::kDefaultMetadataPoolBytes,
												 /*gpu_unit_bytes=*/region_bytes, /*compaction_policy=*/nullptr, /*coordinator_config=*/{},
												 gpu::AllocationPolicy::Pooled);

	std::vector<std::byte> data_dirty(kPayloadBytes, std::byte{0x11});
	std::vector<std::byte> data_clean(kPayloadBytes, std::byte{0x22});
	HostRegionView host_dirty{data_dirty.data(), kPayloadBytes, kSubregionBytes};
	HostRegionView host_clean{data_clean.data(), kPayloadBytes, kSubregionBytes};
	adapter.addRegion(1, host_dirty);  // will be marked dirty before eviction
	adapter.addRegion(2, host_clean);  // will NOT be marked dirty before eviction
	controller.registerRegion(1, host_dirty);
	controller.registerRegion(2, host_clean);

	float v1 = 1.0f, v2 = 2.0f, v3 = 3.0f;
	adapter.next_touched = RegionFootprint{{1}};
	ASSERT_TRUE(controller.insert(MakeRecord(201, v1)).ok);
	adapter.next_touched = RegionFootprint{{2}};
	ASSERT_TRUE(controller.insert(MakeRecord(202, v2)).ok);  // budget now full
	controller.waitIdle();  // promotion is now lazy -- wait for the Coordinator to catch up

	// Simulate a write kernel having touched both regions' device payloads,
	// but only having atomicOr'd a dirty bit for region 1.
	std::vector<std::byte> poison(kPayloadBytes, std::byte{0xFF});
	PokeDevice(controller, /*region=*/1, /*offset=*/header_bytes, poison);
	PokeDevice(controller, /*region=*/2, /*offset=*/header_bytes, poison);
	std::vector<std::byte> dirty_word(gpu::kDirtyWordBytes, std::byte{0});
	dirty_word[0] = std::byte{0x01};  // subregion 0's bit
	PokeDevice(controller, /*region=*/1, /*offset=*/0, dirty_word);
	// region 2's header is left all-zero -- never marked dirty.

	// A 3rd region, same size, forces evicting region 1 (FIFO-oldest).
	std::vector<std::byte> data_c(kPayloadBytes, std::byte{0x33});
	HostRegionView host_c{data_c.data(), kPayloadBytes, kSubregionBytes};
	adapter.addRegion(3, host_c);
	controller.registerRegion(3, host_c);
	adapter.next_touched = RegionFootprint{{3}};
	ASSERT_TRUE(controller.insert(MakeRecord(203, v3)).ok);
	controller.waitIdle();  // eviction/promotion is now lazy -- wait for the Coordinator to catch up

	EXPECT_FALSE(controller.acquireRegion(1).on_device);  // evicted
	EXPECT_TRUE(controller.acquireRegion(2).on_device);   // still resident -- one eviction was enough
	EXPECT_TRUE(controller.acquireRegion(3).on_device);

	// Region 1 was dirty -> its poisoned device payload should have been
	// written back to its host buffer.
	EXPECT_EQ(data_dirty, poison);

	// A 4th region now forces evicting region 2 (the next FIFO-oldest) --
	// its device payload was poisoned exactly like region 1's, but it was
	// never marked dirty, so its host buffer must come back untouched.
	std::vector<std::byte> data_e(kPayloadBytes, std::byte{0x44});
	HostRegionView host_e{data_e.data(), kPayloadBytes, kSubregionBytes};
	adapter.addRegion(4, host_e);
	controller.registerRegion(4, host_e);
	adapter.next_touched = RegionFootprint{{4}};
	ASSERT_TRUE(controller.insert(MakeRecord(204, v3)).ok);
	controller.waitIdle();  // eviction is now lazy -- wait for the Coordinator to catch up

	EXPECT_FALSE(controller.acquireRegion(2).on_device);  // evicted
	EXPECT_TRUE(controller.acquireRegion(3).on_device);
	EXPECT_TRUE(controller.acquireRegion(4).on_device);
	EXPECT_EQ(data_clean, std::vector<std::byte>(kPayloadBytes, std::byte{0x22}));  // untouched: never dirty
}

TEST(ControllerGpuResidencyTest, EvictionAlwaysWritesBackWhenNoHeaderExists) {
	// subregion_bytes == 0 means no dirty header at all -- Arachne can't
	// distinguish dirty from clean, so it must conservatively always write
	// back (see writeBackDirtyRegions()'s doc comment).
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	// 256-byte regions, budget for exactly one at a time -- gpu_unit_bytes
	// below is set to kBytes, so each region occupies exactly one arena unit.
	constexpr std::size_t kBytes = 256;
	Controller controller(adapter, routing_cache, SchedulingConfig{}, nullptr, /*gpu_data_budget_bytes=*/kBytes,
												 gpu::kDefaultMetadataPoolBytes, /*gpu_unit_bytes=*/kBytes, /*compaction_policy=*/nullptr,
												 /*coordinator_config=*/{}, gpu::AllocationPolicy::Pooled);

	std::vector<std::byte> data_a(kBytes, std::byte{0x11});
	std::vector<std::byte> data_b(kBytes, std::byte{0x22});
	HostRegionView host_a{data_a.data(), kBytes, /*subregion_bytes=*/0};
	HostRegionView host_b{data_b.data(), kBytes, /*subregion_bytes=*/0};
	adapter.addRegion(1, host_a);
	adapter.addRegion(2, host_b);
	controller.registerRegion(1, host_a);
	controller.registerRegion(2, host_b);

	float v1 = 1.0f, v2 = 2.0f;
	adapter.next_touched = RegionFootprint{{1}};
	ASSERT_TRUE(controller.insert(MakeRecord(301, v1)).ok);  // budget now full (1 region fits)
	controller.waitIdle();  // promotion is now lazy -- wait for the Coordinator to catch up

	std::vector<std::byte> poison(kBytes, std::byte{0xFF});
	PokeDevice(controller, /*region=*/1, /*offset=*/0, poison);  // never touch a dirty bit -- there is none

	adapter.next_touched = RegionFootprint{{2}};
	ASSERT_TRUE(controller.insert(MakeRecord(302, v2)).ok);  // evicts region 1
	controller.waitIdle();  // eviction is now lazy -- wait for the Coordinator to catch up

	EXPECT_FALSE(controller.acquireRegion(1).on_device);
	EXPECT_EQ(data_a, poison);  // written back unconditionally, despite no dirty bit ever being set
}

TEST(ControllerGpuResidencyTest, EvictionBatchesMultipleRegionsFromOneAnchorInOneCall) {
	// One anchor (minted fresh for record 401's insert-traverse -- see
	// MintAnchorId(), not id 401 itself) depends on two regions at once;
	// evicting it reclaims both in a single evictAnchor() call, exercising
	// writeBackDirtyRegions()'s n > 1 batched-gather path. Region 1 is dirty,
	// region 2 is clean -- both must still be correctly distinguished within
	// the same batch.
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kPayloadBytes = 248;
	constexpr std::size_t kSubregionBytes = 124;
	const std::size_t header_bytes = gpu::DirtyHeaderBytes(kPayloadBytes, kSubregionBytes);
	const std::size_t region_bytes = header_bytes + kPayloadBytes;
	ASSERT_EQ(region_bytes, 256u);

	// Budget for regions 1+2 together (one anchor's whole footprint), plus
	// region 3 needs its own room too -- forces evicting anchor 401
	// entirely (both its regions) before region 3 fits.
	Controller controller(adapter, routing_cache, SchedulingConfig{}, nullptr,
												 /*gpu_data_budget_bytes=*/2 * region_bytes, gpu::kDefaultMetadataPoolBytes,
												 /*gpu_unit_bytes=*/region_bytes, /*compaction_policy=*/nullptr, /*coordinator_config=*/{},
												 gpu::AllocationPolicy::Pooled);

	std::vector<std::byte> data_dirty(kPayloadBytes, std::byte{0x11});
	std::vector<std::byte> data_clean(kPayloadBytes, std::byte{0x22});
	std::vector<std::byte> data_c(kPayloadBytes, std::byte{0x33});
	HostRegionView host_dirty{data_dirty.data(), kPayloadBytes, kSubregionBytes};
	HostRegionView host_clean{data_clean.data(), kPayloadBytes, kSubregionBytes};
	HostRegionView host_c{data_c.data(), kPayloadBytes, kSubregionBytes};
	adapter.addRegion(1, host_dirty);
	adapter.addRegion(2, host_clean);
	adapter.addRegion(3, host_c);
	controller.registerRegion(1, host_dirty);
	controller.registerRegion(2, host_clean);
	controller.registerRegion(3, host_c);

	// A single insert whose footprint covers both regions 1 and 2 -- record
	// 401's insert-traverse anchor now depends on both at once.
	float v1 = 1.0f, v3 = 3.0f;
	adapter.next_touched = RegionFootprint{{1, 2}};
	ASSERT_TRUE(controller.insert(MakeRecord(401, v1)).ok);
	controller.waitIdle();  // promotion is now lazy -- wait for the Coordinator to catch up
	ASSERT_TRUE(controller.acquireRegion(1).on_device);
	ASSERT_TRUE(controller.acquireRegion(2).on_device);

	std::vector<std::byte> poison(kPayloadBytes, std::byte{0xFF});
	PokeDevice(controller, /*region=*/1, /*offset=*/header_bytes, poison);
	PokeDevice(controller, /*region=*/2, /*offset=*/header_bytes, poison);
	std::vector<std::byte> dirty_word(gpu::kDirtyWordBytes, std::byte{0});
	dirty_word[0] = std::byte{0x01};
	PokeDevice(controller, /*region=*/1, /*offset=*/0, dirty_word);  // only region 1 marked dirty

	// Region 3 forces evicting record 401's anchor -- both region 1 and
	// region 2 are reclaimed together, in one evictAnchor() call.
	adapter.next_touched = RegionFootprint{{3}};
	ASSERT_TRUE(controller.insert(MakeRecord(402, v3)).ok);
	controller.waitIdle();  // eviction is now lazy -- wait for the Coordinator to catch up

	EXPECT_FALSE(controller.acquireRegion(1).on_device);
	EXPECT_FALSE(controller.acquireRegion(2).on_device);
	EXPECT_TRUE(controller.acquireRegion(3).on_device);

	EXPECT_EQ(data_dirty, poison);                                              // written back: was dirty
	EXPECT_EQ(data_clean, std::vector<std::byte>(kPayloadBytes, std::byte{0x22}));  // untouched: never dirty
}

TEST(ControllerGpuResidencyTest, InsertRejectsDuplicateIdWithoutTouchingExistingState) {
	FakeAdapter adapter;
	FakeRoutingCache routing_cache;
	constexpr std::size_t kBytes = 256;
	Controller controller(adapter, routing_cache, SchedulingConfig{}, nullptr, /*gpu_data_budget_bytes=*/kBytes,
												 gpu::kDefaultMetadataPoolBytes, /*gpu_unit_bytes=*/kBytes, /*compaction_policy=*/nullptr,
												 /*coordinator_config=*/{}, gpu::AllocationPolicy::Pooled);

	std::vector<std::byte> host_data(kBytes, std::byte{0x42});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	controller.registerRegion(1, host);

	adapter.next_touched = RegionFootprint{{1}};
	float v1 = 1.0f, v2 = 2.0f;
	ASSERT_TRUE(controller.insert(MakeRecord(501, v1)).ok);
	controller.waitIdle();  // promotion is now lazy -- wait for the Coordinator to catch up
	ASSERT_TRUE(controller.acquireRegion(1).on_device);

	// Same id again, different vector -- must be rejected outright, and must
	// not disturb the Region the first insert already promoted.
	InsertResult duplicate = controller.insert(MakeRecord(501, v2));
	EXPECT_FALSE(duplicate.ok);
	EXPECT_TRUE(controller.acquireRegion(1).on_device);

	// remove() frees the id back up for reuse.
	ASSERT_TRUE(controller.remove(501).ok);
	controller.waitIdle();  // release is now lazy -- wait for the Coordinator to catch up
	adapter.next_touched = RegionFootprint{{1}};
	EXPECT_TRUE(controller.insert(MakeRecord(501, v2)).ok);
}

// -----------------------------------------------------------------------------
// Tests for the Anchor-id/VectorId decoupling: MintAnchorId() gives every
// Anchor -- whether traversed-to from insert()'s own lookup or from
// search()'s Hybrid routing -- a fresh id carrying the reserved top bit,
// rather than insert() reusing the about-to-be-inserted record's own id (the
// previous behavior this replaces). And remove() no longer assumes the
// deleted VectorId doubles as some Anchor's id, so it must not reach into
// RegionManager/ReplacementPolicy/RoutingCache at all anymore. See
// MintAnchorId()'s and next_anchor_id_'s doc comments (controller.hpp),
// commitRemove()'s doc comment (controller.cpp), and
// cpp/test/index/report/ for the investigation this closes out.
// -----------------------------------------------------------------------------

TEST(ControllerAnchorIdentityTest, InsertMintsAFreshAnchorIdInsteadOfReusingTheRecordId) {
	FakeAdapter adapter;
	RecordingRoutingCache routing_cache;
	Controller controller(adapter, routing_cache);

	constexpr std::size_t kBytes = 64;
	std::vector<std::byte> host_data(kBytes, std::byte{0x42});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	controller.registerRegion(1, host);

	adapter.next_touched = RegionFootprint{{1}};
	float v1 = 1.0f;
	constexpr VectorId kRecordId = 100;
	ASSERT_TRUE(controller.insert(MakeRecord(kRecordId, v1)).ok);
	controller.waitIdle();  // promotion (and the routing_cache.ensure() it drives) is lazy

	// requestPromotion() -> RoutingCache::ensure() is where the actual anchor
	// id used for this insert's lookup-traverse becomes observable.
	ASSERT_FALSE(routing_cache.ensured_ids.empty());
	for (VectorId anchor_id : routing_cache.ensured_ids) {
		EXPECT_NE(anchor_id, kRecordId) << "insert() must not reuse the inserted record's own id as its Anchor id";
		EXPECT_EQ(anchor_id & kAnchorIdTopBit, kAnchorIdTopBit)
				<< "every minted Anchor id must carry the reserved top bit";
	}
}

TEST(ControllerAnchorIdentityTest, SearchMintsAnAnchorIdWithTheSameReservedTopBitAsInsert) {
	FakeAdapter adapter;
	RecordingRoutingCache routing_cache;
	Controller controller(adapter, routing_cache);

	constexpr std::size_t kBytes = 64;
	std::vector<std::byte> host_data(kBytes, std::byte{0x99});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	controller.registerRegion(1, host);

	adapter.next_touched = RegionFootprint{{1}};
	float query_value = 7.0f;
	Query query{VectorView{&query_value, /*dim=*/1, VectorDType::Float32}, /*top_k=*/1};
	controller.search(query);
	controller.waitIdle();  // requestPromotion() (and its routing_cache.ensure()) is driven off the same worker

	ASSERT_FALSE(routing_cache.ensured_ids.empty());
	for (VectorId anchor_id : routing_cache.ensured_ids) {
		EXPECT_EQ(anchor_id & kAnchorIdTopBit, kAnchorIdTopBit)
				<< "search()'s minted Anchor id must carry the same reserved top bit insert()'s does";
	}
}

TEST(ControllerAnchorIdentityTest, RemoveNeverTouchesReplacementPolicyOrRoutingCacheAnymore) {
	FakeAdapter adapter;
	RecordingRoutingCache routing_cache;
	auto policy = std::make_unique<EvictionCountingReplacementPolicy>();
	EvictionCountingReplacementPolicy* policy_ptr = policy.get();
	// Generous budget -- nothing should ever be evicted here for capacity
	// reasons, so any onAnchorEvicted() call observed can only be explained
	// by remove() itself still reaching into the ReplacementPolicy.
	Controller controller(adapter, routing_cache, SchedulingConfig{}, std::move(policy));

	constexpr std::size_t kBytes = 64;
	std::vector<std::byte> host_data(kBytes, std::byte{0x11});
	HostRegionView host{host_data.data(), kBytes, 0};
	adapter.addRegion(1, host);
	controller.registerRegion(1, host);

	adapter.next_touched = RegionFootprint{{1}};
	float v1 = 1.0f;
	constexpr VectorId kRecordId = 900;
	ASSERT_TRUE(controller.insert(MakeRecord(kRecordId, v1)).ok);
	controller.waitIdle();
	ASSERT_TRUE(controller.acquireRegion(1).on_device);
	routing_cache.erased_ids.clear();  // only care about what remove() itself does below

	ASSERT_TRUE(controller.remove(kRecordId).ok);
	controller.waitIdle();  // release, if any happened, is lazy -- wait for the Coordinator to catch up

	EXPECT_EQ(policy_ptr->eviction_calls, 0)
			<< "commitRemove() must not call region_manager_.releaseAnchor() anymore -- deleting a data vector no "
				 "longer assumes it doubles as some Anchor's id (see commitRemove()'s doc comment)";
	EXPECT_TRUE(routing_cache.erased_ids.empty()) << "remove() must not erase anything from the RoutingCache either";
	// The Region the insert promoted is untouched by the delete -- its
	// Anchor's own lifecycle (heat/capacity-driven) is unaffected by whether
	// the data vector that originally caused it to exist is still live.
	EXPECT_TRUE(controller.acquireRegion(1).on_device);
}

}  // namespace
