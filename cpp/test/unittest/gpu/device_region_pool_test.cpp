#include "gpu/device_region_pool.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpu/compaction_policy.hpp"
#include "gpu/device_context.hpp"

// Tests for DeviceRegionPool (gpu/device_region_pool.hpp / .cpp), organized
// into four groups:
//
//  1. DeviceRegionPoolPolicyTest (INSTANTIATE_TEST_SUITE_P'd as
//     "PooledAndNormal") -- the core allocate/free/copy/lease contract,
//     run once under each AllocationPolicy. Both policies must satisfy the
//     exact same observable behavior; nothing here is Pooled- or
//     Normal-specific except the two acquire()/free() timing tests, which
//     exercise Lease-based cross-thread/cross-stream synchronization that
//     applies identically under either policy.
//
//  2. DeviceContextStreamTest and standalone DeviceRegionPoolTest cases --
//     DeviceContext-level concerns (worker stream count/identity, default
//     policy, arena reservation and unit-rounding) that don't need a live
//     DeviceRegionPool.
//
//  3. DeviceRegionPoolCompactionTest -- fragmentation-triggered compaction.
//     Pooled-only, since Normal has no shared arena to fragment. Its
//     SetUp() builds one canonical layout reused by every test in the
//     fixture: a 10-unit arena filled with five 2-unit allocations, then
//     every other one freed, leaving three isolated 2-unit holes that
//     individually can't satisfy a 4-unit request despite 6 units being
//     free in aggregate:
//
//       unit:    0    1    2    3    4    5    6    7    8    9
//              +----+----+----+----+----+----+----+----+----+----+
//              | free    | P1 (live)| free    | P3 (live)| free    |
//              +----+----+----+----+----+----+----+----+----+----+
//                P0         P1         P2         P3         P4
//
//     P0/P2/P4 are freed in SetUp(); P1/P3 survive and their data must stay
//     intact across whatever relocation compaction performs. This is the
//     textbook external-fragmentation shape the self-healing
//     allocate()/tryAllocate() and explicit compact() exist to resolve.
//
//  4. DeviceRegionPoolStressTest -- thousands of rounds of random
//     allocate/free churn (RunAllocDeallocChurn), keeping live bytes under
//     a fixed cap so accounting never genuinely runs out while still
//     routinely fragmenting the Pooled arena. Verifies every survivor's
//     byte pattern (VerifyLiveData) stays correct across many compaction
//     events, not just one hand-constructed one.

namespace {

using arachne::gpu::AllocationPolicy;
using arachne::gpu::DeviceContext;
using arachne::gpu::kDefaultDataPoolBytes;
using arachne::gpu::kDefaultMetadataPoolBytes;
using arachne::gpu::MemoryKind;
using arachne::gpu::DeviceRegionHandle;
using arachne::gpu::DeviceRegionPool;
using arachne::gpu::NoCompactionPolicy;
using arachne::gpu::TargetedCompactionPolicy;

// Parametrized over AllocationPolicy: both alternatives must satisfy the
// exact same DeviceRegionPool contract, since callers (DeviceRegionPool itself
// included) never branch on which one is active -- see DeviceContext's
// AllocationPolicy doc comment.
class DeviceRegionPoolPolicyTest : public ::testing::TestWithParam<AllocationPolicy> {
 protected:
	DeviceContext MakeDevice() { return DeviceContext(/*device_id=*/0, GetParam()); }
};

TEST_P(DeviceRegionPoolPolicyTest, AllocateReturnsValidHandleAndTracksBytes) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(1024);

	EXPECT_TRUE(handle.valid());
	EXPECT_EQ(pool.bytesAllocated(), 1024u);
}

TEST_P(DeviceRegionPoolPolicyTest, CopyRoundTripsThroughDeviceMemory) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	constexpr std::size_t kBytes = 256;
	DeviceRegionHandle handle = pool.allocate(kBytes);

	std::vector<std::byte> host_in(kBytes);
	for (std::size_t i = 0; i < kBytes; ++i) host_in[i] = std::byte{static_cast<unsigned char>(i)};
	pool.copyFromHost(handle, host_in.data(), kBytes);

	std::vector<std::byte> host_out(kBytes);
	pool.copyToHost(handle, host_out.data(), kBytes);

	EXPECT_EQ(host_in, host_out);

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, FreeReclaimsBytesAndInvalidatesHandle) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(512);
	ASSERT_EQ(pool.bytesAllocated(), 512u);

	pool.free(handle);

	EXPECT_EQ(pool.bytesAllocated(), 0u);
	EXPECT_THROW(pool.acquire(handle), std::invalid_argument);
}

TEST_P(DeviceRegionPoolPolicyTest, FreeOfInvalidHandleIsANoop) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	pool.free(DeviceRegionHandle{});  // default-constructed, never allocated

	EXPECT_EQ(pool.bytesAllocated(), 0u);
}

TEST_P(DeviceRegionPoolPolicyTest, MultipleAllocationsAreIndependent) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle a = pool.allocate(128);
	DeviceRegionHandle b = pool.allocate(256);

	EXPECT_NE(a.id, b.id);
	EXPECT_EQ(pool.bytesAllocated(), 384u);
	{
		DeviceRegionPool::Lease lease_a = pool.acquire(a);
		DeviceRegionPool::Lease lease_b = pool.acquire(b);
		EXPECT_NE(lease_a.ptr(), lease_b.ptr());
	}

	pool.free(a);
	EXPECT_EQ(pool.bytesAllocated(), 256u);
	EXPECT_NO_THROW(pool.acquire(b));

	pool.free(b);
}

TEST_P(DeviceRegionPoolPolicyTest, DataAndMetadataAllocationsAreTrackedSeparately) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle data = pool.allocate(1024, MemoryKind::Data);
	DeviceRegionHandle metadata = pool.allocate(128, MemoryKind::Metadata);

	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), 1024u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Metadata), 128u);
	EXPECT_EQ(pool.bytesAllocated(), 1024u + 128u);

	pool.free(data);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), 0u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Metadata), 128u);  // untouched

	pool.free(metadata);
}

TEST_P(DeviceRegionPoolPolicyTest, MetadataAllocationRoundTripsThroughDeviceMemory) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	constexpr std::size_t kBytes = 64;
	DeviceRegionHandle handle = pool.allocate(kBytes, MemoryKind::Metadata);

	std::vector<std::byte> host_in(kBytes, std::byte{0x5a});
	pool.copyFromHost(handle, host_in.data(), kBytes);

	std::vector<std::byte> host_out(kBytes);
	pool.copyToHost(handle, host_out.data(), kBytes);
	EXPECT_EQ(host_in, host_out);

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, CopyFromHostRejectsOversizedRequest) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(16);
	std::vector<std::byte> host_in(32);

	EXPECT_THROW(pool.copyFromHost(handle, host_in.data(), 32), std::out_of_range);

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, CopyFromHostRejectsOffsetPlusBytesExceedingAllocation) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(16);
	std::vector<std::byte> host_in(8);

	// 12 (offset) + 8 (bytes) = 20 > 16 -- fits neither the fast in-range case
	// nor a plain bytes-only check, only the offset-aware one.
	EXPECT_THROW(pool.copyFromHost(handle, host_in.data(), 8, /*dst_offset=*/12), std::out_of_range);

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, CopyRoundTripsWithHeaderOffset) {
	// Mirrors the prepended-dirty-header layout Controller::make() uses: the
	// first kHeaderBytes of the allocation are reserved, and the actual
	// payload starts right after it.
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	constexpr std::size_t kHeaderBytes = 8;
	constexpr std::size_t kPayloadBytes = 256;
	DeviceRegionHandle handle = pool.allocate(kHeaderBytes + kPayloadBytes);

	std::vector<std::byte> header_in(kHeaderBytes, std::byte{0xAA});
	pool.copyFromHost(handle, header_in.data(), kHeaderBytes, /*dst_offset=*/0);

	std::vector<std::byte> payload_in(kPayloadBytes);
	for (std::size_t i = 0; i < kPayloadBytes; ++i) payload_in[i] = std::byte{static_cast<unsigned char>(i)};
	pool.copyFromHost(handle, payload_in.data(), kPayloadBytes, /*dst_offset=*/kHeaderBytes);

	std::vector<std::byte> header_out(kHeaderBytes);
	pool.copyToHost(handle, header_out.data(), kHeaderBytes, /*src_offset=*/0);
	EXPECT_EQ(header_in, header_out);

	std::vector<std::byte> payload_out(kPayloadBytes);
	pool.copyToHost(handle, payload_out.data(), kPayloadBytes, /*src_offset=*/kHeaderBytes);
	EXPECT_EQ(payload_in, payload_out);

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, EnqueueCopyToHostBatchesMultipleHandlesBeforeOneFlush) {
	// Three allocations read back via enqueueCopyToHost() sharing one
	// `pending` and a single flush() -- proves the batched gather path
	// (Controller::writeBackDirtyRegions()) lands every copy correctly, not
	// just the single-handle copyToHost() convenience path.
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	constexpr std::size_t kBytes = 128;
	DeviceRegionHandle a = pool.allocate(kBytes);
	DeviceRegionHandle b = pool.allocate(kBytes);
	DeviceRegionHandle c = pool.allocate(kBytes);

	std::vector<std::byte> in_a(kBytes, std::byte{0xA1});
	std::vector<std::byte> in_b(kBytes, std::byte{0xB2});
	std::vector<std::byte> in_c(kBytes, std::byte{0xC3});
	pool.copyFromHost(a, in_a.data(), kBytes);
	pool.copyFromHost(b, in_b.data(), kBytes);
	pool.copyFromHost(c, in_c.data(), kBytes);

	std::vector<std::byte> out_a(kBytes), out_b(kBytes), out_c(kBytes);
	{
		std::vector<DeviceRegionPool::Lease> pending;
		pool.enqueueCopyToHost(a, out_a.data(), kBytes, /*src_offset=*/0, pending);
		pool.enqueueCopyToHost(b, out_b.data(), kBytes, /*src_offset=*/0, pending);
		pool.enqueueCopyToHost(c, out_c.data(), kBytes, /*src_offset=*/0, pending);
		// Nothing is guaranteed to have landed yet -- only flush() proves it.
		pool.flush();
	}  // `pending`'s Leases release here, after flush() already proved every copy landed

	EXPECT_EQ(in_a, out_a);
	EXPECT_EQ(in_b, out_b);
	EXPECT_EQ(in_c, out_c);

	pool.free(a);
	pool.free(b);
	pool.free(c);
}

TEST_P(DeviceRegionPoolPolicyTest, HasCapacityReflectsBudgetAndOutstandingBytes) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	EXPECT_TRUE(pool.hasCapacity(1024));
	EXPECT_FALSE(pool.hasCapacity(device.budgetBytes(MemoryKind::Data) + 1));

	DeviceRegionHandle handle = pool.allocate(1024);
	EXPECT_TRUE(pool.hasCapacity(device.budgetBytes(MemoryKind::Data) - 1024));
	EXPECT_FALSE(pool.hasCapacity(device.budgetBytes(MemoryKind::Data) - 1024 + 1));

	pool.free(handle);
	EXPECT_TRUE(pool.hasCapacity(device.budgetBytes(MemoryKind::Data)));
}

TEST_P(DeviceRegionPoolPolicyTest, TryAllocateReturnsNulloptWhenOverBudget) {
	// Budget and unit_bytes both deliberately tiny -- data_pool_bytes/
	// unit_bytes == 4 units total, so a request for double that already
	// exceeds it regardless of how much real GPU memory is actually free,
	// and regardless of AllocationPolicy::Pooled's unit-rounding.
	constexpr std::size_t kUnitBytes = 256;
	DeviceContext device(/*device_id=*/0, GetParam(), /*data_pool_bytes=*/4 * kUnitBytes,
											 arachne::gpu::kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, kUnitBytes);
	DeviceRegionPool pool(device);

	std::optional<DeviceRegionHandle> handle = pool.tryAllocate(8 * kUnitBytes);
	EXPECT_FALSE(handle.has_value());
	EXPECT_EQ(pool.bytesAllocated(), 0u);
}

TEST_P(DeviceRegionPoolPolicyTest, TryAllocateSucceedsWithinBudgetAndUpdatesCapacity) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	std::optional<DeviceRegionHandle> handle = pool.tryAllocate(1024);
	ASSERT_TRUE(handle.has_value());
	EXPECT_TRUE(handle->valid());
	EXPECT_EQ(pool.bytesAllocated(), 1024u);

	pool.free(*handle);
}

TEST_P(DeviceRegionPoolPolicyTest, AcquireReturnsLeaseWithMatchingPointer) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(128);
	void* first_ptr = nullptr;
	{
		DeviceRegionPool::Lease lease = pool.acquire(handle);
		first_ptr = lease.ptr();
		EXPECT_NE(first_ptr, nullptr);
	}  // released here
	{
		// A second, separate acquire() (no compaction in between) must resolve
		// to the same pointer.
		DeviceRegionPool::Lease lease = pool.acquire(handle);
		EXPECT_EQ(lease.ptr(), first_ptr);
	}

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, AcquireOfInvalidHandleThrows) {
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	EXPECT_THROW(pool.acquire(DeviceRegionHandle{}), std::invalid_argument);
}

TEST_P(DeviceRegionPoolPolicyTest, FreeWaitsForOutstandingLeaseToRelease) {
	// While a Lease is held (on any thread), free() must not reclaim the
	// memory -- it should block until the Lease is released. Proven here by
	// timing: the holder sleeps before releasing, and free() must not return
	// before that sleep elapses.
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(128);
	constexpr auto kHoldDuration = std::chrono::milliseconds(150);

	std::thread holder([&] {
		DeviceRegionPool::Lease lease = pool.acquire(handle);
		std::this_thread::sleep_for(kHoldDuration);
		// lease released here, at thread exit
	});

	// Give the holder thread a moment to actually acquire before we race it.
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	auto start = std::chrono::steady_clock::now();
	pool.free(handle);
	auto elapsed = std::chrono::steady_clock::now() - start;

	holder.join();
	EXPECT_GE(elapsed, kHoldDuration - std::chrono::milliseconds(20));
}

TEST_P(DeviceRegionPoolPolicyTest, FreeWaitsForAllOutstandingLeasesAcrossDifferentStreams) {
	// Two Leases on the same handle, acquired on two different streams --
	// free() must wait for both, not just the first to release.
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(128);
	cudaStream_t stream_a = nullptr;
	cudaStream_t stream_b = nullptr;
	ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
	ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

	constexpr auto kHoldDuration = std::chrono::milliseconds(150);

	std::thread holder_a([&] {
		DeviceRegionPool::Lease lease = pool.acquire(handle, stream_a);
		std::this_thread::sleep_for(kHoldDuration);
	});
	std::thread holder_b([&] {
		DeviceRegionPool::Lease lease = pool.acquire(handle, stream_b);
		std::this_thread::sleep_for(kHoldDuration);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	auto start = std::chrono::steady_clock::now();
	pool.free(handle);
	auto elapsed = std::chrono::steady_clock::now() - start;

	holder_a.join();
	holder_b.join();
	EXPECT_GE(elapsed, kHoldDuration - std::chrono::milliseconds(20));

	cudaStreamDestroy(stream_a);
	cudaStreamDestroy(stream_b);
}

TEST_P(DeviceRegionPoolPolicyTest, CrossStreamAcquireOrdersAfterPriorStreamsReleasedWork) {
	// Exercises acquire()'s cross-stream event-wait: a Lease released on
	// stream A (write not yet known to land) followed by stream B acquiring
	// the same handle with no host-side sync. Without a GPU-side wait,
	// stream B could read stale bytes; 1 MiB gives the race a real window.
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	constexpr std::size_t kBytes = 1024 * 1024;
	DeviceRegionHandle handle = pool.allocate(kBytes);

	cudaStream_t stream_a = nullptr;
	cudaStream_t stream_b = nullptr;
	ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
	ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

	std::vector<std::byte> pattern(kBytes, std::byte{0xAA});
	{
		DeviceRegionPool::Lease lease = pool.acquire(handle, stream_a);
		ASSERT_EQ(cudaMemcpyAsync(lease.ptr(), pattern.data(), kBytes, cudaMemcpyHostToDevice, stream_a),
							cudaSuccess);
		// `lease` releases here -- only an event is recorded on stream_a, no
		// host wait for the copy to actually finish.
	}

	std::vector<std::byte> out(kBytes);
	{
		DeviceRegionPool::Lease lease = pool.acquire(handle, stream_b);  // different stream, no sync since above
		ASSERT_EQ(cudaMemcpyAsync(out.data(), lease.ptr(), kBytes, cudaMemcpyDeviceToHost, stream_b),
							cudaSuccess);
	}
	ASSERT_EQ(cudaStreamSynchronize(stream_b), cudaSuccess);

	EXPECT_EQ(out, pattern);

	pool.free(handle);
	cudaStreamDestroy(stream_a);
	cudaStreamDestroy(stream_b);
}

INSTANTIATE_TEST_SUITE_P(PooledAndNormal, DeviceRegionPoolPolicyTest,
													::testing::Values(AllocationPolicy::Pooled, AllocationPolicy::Normal),
													[](const ::testing::TestParamInfo<AllocationPolicy>& info) {
														return info.param == AllocationPolicy::Pooled ? "Pooled" : "Normal";
													});

TEST(DeviceRegionPoolTest, DefaultDeviceContextUsesNormalPolicy) {
	// Normal is the no-pooling/no-compaction baseline -- Pooled is an
	// explicit opt-in (see AllocationPolicy's own doc comment), not the
	// default anymore.
	DeviceContext device;
	EXPECT_EQ(device.allocationPolicy(), AllocationPolicy::Normal);
}

TEST(DeviceContextStreamTest, DefaultWorkerStreamCountIsOne) {
	DeviceContext device;
	EXPECT_EQ(device.workerStreamCount(), 1u);
}

TEST(DeviceContextStreamTest, WorkerStreamCountMatchesConstructorParam) {
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, kDefaultDataPoolBytes,
											 kDefaultMetadataPoolBytes, /*worker_stream_count=*/4);
	EXPECT_EQ(device.workerStreamCount(), 4u);
}

TEST(DeviceContextStreamTest, WorkerStreamsAreDistinctFromEachOtherAndFromManagementStream) {
	// Controller relies on this to route promotion/eviction traffic
	// (management stream) and per-worker kernel launches (worker streams)
	// onto genuinely independent CUDA streams.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, kDefaultDataPoolBytes,
											 kDefaultMetadataPoolBytes, /*worker_stream_count=*/4);

	std::vector<cudaStream_t> streams;
	streams.push_back(device.managementStream());
	for (std::size_t i = 0; i < device.workerStreamCount(); ++i) streams.push_back(device.workerStream(i));

	for (std::size_t i = 0; i < streams.size(); ++i) {
		EXPECT_NE(streams[i], nullptr) << "stream " << i;
		for (std::size_t j = i + 1; j < streams.size(); ++j) {
			EXPECT_NE(streams[i], streams[j]) << "streams " << i << " and " << j << " should be distinct";
		}
	}
}

TEST(DeviceContextStreamTest, WorkerStreamOutOfRangeThrows) {
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, kDefaultDataPoolBytes,
											 kDefaultMetadataPoolBytes, /*worker_stream_count=*/2);
	EXPECT_NO_THROW(device.workerStream(1));
	EXPECT_THROW(device.workerStream(2), std::out_of_range);
}

TEST(DeviceRegionPoolTest, NormalPolicyDoesNotPreReserveAnArena) {
	// Normal shouldn't reserve pool bytes up front -- sizes absurd enough to
	// fail construction under Pooled should still succeed here, since Normal
	// ignores them entirely.
	constexpr std::size_t kAbsurdlyLarge = std::size_t{1} << 40;  // 1 TiB
	EXPECT_NO_THROW(DeviceContext(/*device_id=*/0, AllocationPolicy::Normal, kAbsurdlyLarge, kAbsurdlyLarge));
}

TEST(DeviceRegionPoolTest, PooledBudgetBytesReflectsUnitRoundedArenaCapacity) {
	constexpr std::size_t kUnitBytes = 256;
	// 1000 is not a multiple of 256 -- rounds up to 4 units == 1024 bytes.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, /*data_pool_bytes=*/1000,
											 kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, kUnitBytes);
	EXPECT_EQ(device.budgetBytes(MemoryKind::Data), 4 * kUnitBytes);
}

// ---------------------------------------------------------------------------
// Fragmentation-triggered compaction (Pooled-only, see file-level comment
// above for the canonical layout SetUp() builds).
// ---------------------------------------------------------------------------

class DeviceRegionPoolCompactionTest : public ::testing::Test {
 protected:
	static constexpr std::size_t kUnitBytes = 128;
	static constexpr std::size_t kTotalUnits = 10;

	DeviceContext device_{/*device_id=*/0,
												 AllocationPolicy::Pooled,
												 kUnitBytes * kTotalUnits,
												 kDefaultMetadataPoolBytes,
												 /*worker_stream_count=*/1,
												 kUnitBytes};
	DeviceRegionPool pool_{device_};

	// See file-level comment above for the exact layout this SetUp() builds
	// (5 pieces, every other one freed).
	std::vector<DeviceRegionHandle> pieces_;
	std::vector<std::byte> patterns_;

	void SetUp() override {
		for (int i = 0; i < 5; ++i) {
			DeviceRegionHandle handle = pool_.allocate(2 * kUnitBytes);
			std::byte pattern{static_cast<unsigned char>(0x10 + i)};
			std::vector<std::byte> data(2 * kUnitBytes, pattern);
			pool_.copyFromHost(handle, data.data(), data.size());
			pieces_.push_back(handle);
			patterns_.push_back(pattern);
		}
		pool_.free(pieces_[0]);
		pool_.free(pieces_[2]);
		pool_.free(pieces_[4]);
	}

	void ExpectSurvivorsIntact() {
		for (std::size_t idx : {1u, 3u}) {
			std::vector<std::byte> out(2 * kUnitBytes);
			pool_.copyToHost(pieces_[idx], out.data(), out.size());
			for (std::byte b : out) EXPECT_EQ(b, patterns_[idx]);
		}
	}
};

TEST_F(DeviceRegionPoolCompactionTest, TryAllocateSelfHealsThroughFragmentation) {
	// Best-fit alone can't satisfy this (largest hole is 2 units, this needs
	// 4) -- tryAllocate() must succeed anyway, by relocating one of the two
	// survivors internally before retrying.
	std::optional<DeviceRegionHandle> handle = pool_.tryAllocate(4 * kUnitBytes);
	ASSERT_TRUE(handle.has_value());
	ExpectSurvivorsIntact();

	std::vector<std::byte> out(4 * kUnitBytes);
	pool_.copyToHost(*handle, out.data(), out.size());  // must be readable/valid, not garbage

	pool_.free(*handle);
	pool_.free(pieces_[1]);
	pool_.free(pieces_[3]);
}

TEST_F(DeviceRegionPoolCompactionTest, AllocateThrowsWhenGenuinelyOutOfBudgetDespiteFragmentation) {
	// 12 units' worth on a 10-unit arena -- no relocation plan, however
	// clever, can conjure capacity that was never there.
	EXPECT_THROW(pool_.allocate(12 * kUnitBytes), std::runtime_error);
}

TEST_F(DeviceRegionPoolCompactionTest, CompactSatisfiesAFutureAllocateOfTheGivenSize) {
	DeviceRegionPool::CompactionResult result = pool_.compact(MemoryKind::Data, 4 * kUnitBytes);
	EXPECT_GT(result.relocated_count, 0u);
	ExpectSurvivorsIntact();

	// Now that compact() opened the room, an ordinary allocate() (no
	// internal retry needed) must succeed without throwing.
	DeviceRegionHandle handle = pool_.allocate(4 * kUnitBytes);
	pool_.free(handle);
	pool_.free(pieces_[1]);
	pool_.free(pieces_[3]);
}

TEST_F(DeviceRegionPoolCompactionTest, CompactIsANoopWhenAlreadySatisfiable) {
	// 2 units is already satisfiable by any one of the three existing holes
	// -- nothing should move.
	DeviceRegionPool::CompactionResult result = pool_.compact(MemoryKind::Data, 2 * kUnitBytes);
	EXPECT_EQ(result.relocated_count, 0u);
	EXPECT_EQ(result.bytes_relocated, 0u);
}

TEST_F(DeviceRegionPoolCompactionTest, CompactDoesNotWaitForAPinnedAllocationAndWorksAroundIt) {
	// Pin piece[1] behind a held Lease on a background thread -- compact()
	// must not block on it, but can still succeed by relocating piece[3]
	// (the only remaining unpinned candidate) instead. Proven by timing.
	constexpr auto kHoldDuration = std::chrono::milliseconds(150);
	std::thread holder([&] {
		DeviceRegionPool::Lease lease = pool_.acquire(pieces_[1]);
		std::this_thread::sleep_for(kHoldDuration);
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let the holder actually acquire

	auto start = std::chrono::steady_clock::now();
	DeviceRegionPool::CompactionResult result = pool_.compact(MemoryKind::Data, 4 * kUnitBytes);
	auto elapsed = std::chrono::steady_clock::now() - start;

	EXPECT_LT(elapsed, kHoldDuration);  // did not wait for piece[1]'s Lease
	EXPECT_EQ(result.relocated_count, 1u);  // used piece[3] instead

	holder.join();
	ExpectSurvivorsIntact();
}

TEST_F(DeviceRegionPoolCompactionTest, NoCompactionPolicyMakesFragmentationFatal) {
	// A dedicated DeviceContext/DeviceRegionPool pair (independent of the
	// fixture's own device_/pool_) reproduces the same fragmented layout,
	// but with NoCompactionPolicy injected instead of the default.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, kUnitBytes * kTotalUnits,
											 kDefaultMetadataPoolBytes, /*worker_stream_count=*/1, kUnitBytes);
	DeviceRegionPool isolated(device, std::make_unique<NoCompactionPolicy>());

	std::vector<DeviceRegionHandle> handles;
	for (int i = 0; i < 5; ++i) handles.push_back(isolated.allocate(2 * kUnitBytes));
	isolated.free(handles[0]);
	isolated.free(handles[2]);
	isolated.free(handles[4]);

	// NoCompactionPolicy opts this pool out of relocation entirely --
	// fragmentation TargetedCompactionPolicy would resolve is now a hard
	// failure, like Normal's own allocator under real memory pressure.
	EXPECT_THROW(isolated.allocate(4 * kUnitBytes), std::runtime_error);
	EXPECT_EQ(isolated.compact(MemoryKind::Data, 4 * kUnitBytes).relocated_count, 0u);

	isolated.free(handles[1]);
	isolated.free(handles[3]);
}

// ---------------------------------------------------------------------------
// Stress tests (see file-level comment above): random churn keeping live
// bytes under budget while routinely fragmenting the Pooled arena.
// ---------------------------------------------------------------------------

struct LiveEntry {
	DeviceRegionHandle handle;
	std::size_t bytes;
	std::byte canary;
};

// Runs `iterations` rounds of random allocate-with-data (byte pattern
// written and later checked, to catch corruption from compaction moves) or
// free, keeping live footprint under `max_live_bytes`. Returns whatever is
// still live at the end -- the caller frees it.
std::vector<LiveEntry> RunAllocDeallocChurn(DeviceRegionPool& pool, std::mt19937& rng, int iterations,
																						 std::size_t max_live_bytes, std::size_t min_bytes = 1024,
																						 std::size_t max_bytes = 1024 * 1024) {
	std::uniform_int_distribution<std::size_t> size_dist(min_bytes, max_bytes);
	std::vector<LiveEntry> live;
	std::size_t live_bytes = 0;

	for (int i = 0; i < iterations; ++i) {
		bool should_allocate = live.empty() || (live_bytes < max_live_bytes && (rng() % 3 != 0));
		if (should_allocate) {
			std::size_t bytes = size_dist(rng);
			if (live_bytes + bytes > max_live_bytes) continue;

			DeviceRegionHandle handle = pool.allocate(bytes, MemoryKind::Data);
			std::byte canary = std::byte{static_cast<unsigned char>(rng() & 0xFF)};
			std::vector<std::byte> pattern(bytes, canary);
			pool.copyFromHost(handle, pattern.data(), bytes);

			live.push_back(LiveEntry{handle, bytes, canary});
			live_bytes += bytes;
		} else {
			std::size_t idx = rng() % live.size();
			live_bytes -= live[idx].bytes;
			pool.free(live[idx].handle);
			live.erase(live.begin() + idx);
		}
	}
	return live;
}

void VerifyLiveData(DeviceRegionPool& pool, const std::vector<LiveEntry>& live) {
	for (const LiveEntry& entry : live) {
		std::vector<std::byte> out(entry.bytes);
		pool.copyToHost(entry.handle, out.data(), entry.bytes);
		for (std::byte b : out) {
			ASSERT_EQ(b, entry.canary);
		}
	}
}

TEST(DeviceRegionPoolStressTest, PooledPolicySurvivesExcessiveAllocDeallocAndCompaction) {
	cudaSetDevice(0);
	std::size_t free_bytes = 0;
	std::size_t total_bytes = 0;
	ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);

	// The 90% figure proves Arachne can reserve a big-chunk arena at that
	// scale, not that it churns gigabytes of live data every run -- see
	// kMaxLiveBytes below for the actual churn volume.
	std::size_t budget = static_cast<std::size_t>(static_cast<double>(free_bytes) * 0.9);
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, budget, kDefaultMetadataPoolBytes);
	DeviceRegionPool pool(device);

	std::mt19937 rng(12345);
	constexpr std::size_t kMaxLiveBytes = 128 * 1024 * 1024;  // 128 MiB
	constexpr int kIterations = 3000;

	std::vector<LiveEntry> live = RunAllocDeallocChurn(pool, rng, kIterations, kMaxLiveBytes);
	ASSERT_FALSE(live.empty());
	VerifyLiveData(pool, live);

	std::size_t before = pool.bytesAllocated(MemoryKind::Data);
	DeviceRegionPool::CompactionResult result = pool.compact(MemoryKind::Data, kMaxLiveBytes);
	std::size_t after = pool.bytesAllocated(MemoryKind::Data);

	EXPECT_EQ(before, after);  // compaction changes addresses, not live totals
	(void)result;              // how much (if anything) actually needed moving depends on the
															// random churn's own fragmentation -- not asserted on directly

	VerifyLiveData(pool, live);  // data must survive whatever compact() did

	for (const LiveEntry& entry : live) pool.free(entry.handle);
}

TEST(DeviceRegionPoolStressTest, NormalPolicySurvivesExcessiveAllocDealloc) {
	// Normal goes through RAFT/RMM's cuda_memory_resource for every single
	// allocate()/free() (real cudaMalloc/cudaFree, no pooling) -- exercises
	// that path under the same churn as the Pooled stress test above.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Normal);
	DeviceRegionPool pool(device);

	std::mt19937 rng(54321);
	constexpr std::size_t kMaxLiveBytes = 64 * 1024 * 1024;  // 64 MiB
	constexpr int kIterations = 2000;

	std::vector<LiveEntry> live = RunAllocDeallocChurn(pool, rng, kIterations, kMaxLiveBytes);
	ASSERT_FALSE(live.empty());
	VerifyLiveData(pool, live);

	// compact() is a documented no-op under Normal (no shared arena to
	// consolidate) -- confirm that explicitly here rather than only relying
	// on the parametrized CompactionTest cases above.
	std::size_t before = pool.bytesAllocated(MemoryKind::Data);
	DeviceRegionPool::CompactionResult result = pool.compact(MemoryKind::Data, kMaxLiveBytes);

	EXPECT_EQ(result.relocated_count, 0u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), before);
	VerifyLiveData(pool, live);  // untouched, as expected

	for (const LiveEntry& entry : live) pool.free(entry.handle);
}

TEST(DeviceRegionPoolStressTest, PooledPolicySelfHealsThroughHeavyFragmentationUnderATightBudget) {
	// Unlike the huge-budget stress tests above, this uses a small unit size
	// and a budget close to the live-data cap so nearly every allocate()
	// during the churn below has a real chance of hitting a fragmented arena
	// and needing tryAllocate()'s self-healing compaction. Budget still has
	// 2x headroom so unit-rounding waste alone can't cause a spurious OOM.
	constexpr std::size_t kUnitBytes = 4 * 1024;             // 4 KiB
	constexpr std::size_t kMaxLiveBytes = 2 * 1024 * 1024;   // 2 MiB
	constexpr std::size_t kBudget = 4 * 1024 * 1024;         // 4 MiB -- 2x headroom
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, kBudget, kDefaultMetadataPoolBytes,
											 /*worker_stream_count=*/1, kUnitBytes);
	DeviceRegionPool pool(device, std::make_unique<TargetedCompactionPolicy>());

	std::mt19937 rng(999331);
	constexpr int kIterations = 5000;

	std::vector<LiveEntry> live =
			RunAllocDeallocChurn(pool, rng, kIterations, kMaxLiveBytes, /*min_bytes=*/kUnitBytes / 2,
														/*max_bytes=*/16 * kUnitBytes);
	ASSERT_FALSE(live.empty());
	VerifyLiveData(pool, live);  // every survivor's data is intact, however many times it moved

	for (const LiveEntry& entry : live) pool.free(entry.handle);

	// The whole budget must be reclaimable after everything is freed --
	// proves free()'s coalescing isn't leaking free-extent bookkeeping
	// across thousands of allocate()/free()/relocate() cycles.
	DeviceRegionHandle whole = pool.allocate(kBudget);
	pool.free(whole);
}

}  // namespace
