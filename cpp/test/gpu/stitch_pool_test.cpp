#include "gpu/stitch_pool.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "gpu/device_context.hpp"

namespace {

using arachne::gpu::AccessMode;
using arachne::gpu::AllocationPolicy;
using arachne::gpu::DeviceContext;
using arachne::gpu::kDefaultMetadataPoolBytes;
using arachne::gpu::MemoryKind;
using arachne::gpu::StitchHandle;
using arachne::gpu::StitchPool;

// Parametrized over AllocationPolicy: both alternatives must satisfy the
// exact same StitchPool contract, since callers (StitchPool itself
// included) never branch on which one is active -- see DeviceContext's
// AllocationPolicy doc comment.
class StitchPoolPolicyTest : public ::testing::TestWithParam<AllocationPolicy> {
 protected:
	DeviceContext MakeDevice() { return DeviceContext(/*device_id=*/0, GetParam()); }
};

TEST_P(StitchPoolPolicyTest, AllocateReturnsValidHandleAndTracksBytes) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(1024);

	EXPECT_TRUE(handle.valid());
	EXPECT_EQ(pool.bytesAllocated(), 1024u);
}

TEST_P(StitchPoolPolicyTest, CopyRoundTripsThroughDeviceMemory) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	constexpr std::size_t kBytes = 256;
	StitchHandle handle = pool.allocate(kBytes);

	std::vector<std::byte> host_in(kBytes);
	for (std::size_t i = 0; i < kBytes; ++i) host_in[i] = std::byte{static_cast<unsigned char>(i)};
	pool.copyFromHost(handle, host_in.data(), kBytes);

	std::vector<std::byte> host_out(kBytes);
	pool.copyToHost(handle, host_out.data(), kBytes);

	EXPECT_EQ(host_in, host_out);

	pool.free(handle);
}

TEST_P(StitchPoolPolicyTest, FreeReclaimsBytesAndInvalidatesHandle) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(512);
	ASSERT_EQ(pool.bytesAllocated(), 512u);

	pool.free(handle);

	EXPECT_EQ(pool.bytesAllocated(), 0u);
	EXPECT_THROW(pool.access(handle, AccessMode::Read), std::invalid_argument);
}

TEST_P(StitchPoolPolicyTest, FreeOfInvalidHandleIsANoop) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	pool.free(StitchHandle{});  // default-constructed, never allocated

	EXPECT_EQ(pool.bytesAllocated(), 0u);
}

TEST_P(StitchPoolPolicyTest, AccessOfNeverAllocatedHandleThrows) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	EXPECT_THROW(pool.access(StitchHandle{}, AccessMode::Read), std::invalid_argument);
}

TEST_P(StitchPoolPolicyTest, MultipleAllocationsAreIndependent) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle a = pool.allocate(128);
	StitchHandle b = pool.allocate(256);

	EXPECT_NE(a.id, b.id);
	EXPECT_EQ(pool.bytesAllocated(), 384u);
	EXPECT_NE(pool.access(a, AccessMode::Read), pool.access(b, AccessMode::Read));

	pool.free(a);
	EXPECT_EQ(pool.bytesAllocated(), 256u);
	EXPECT_NO_THROW(pool.access(b, AccessMode::Read));

	pool.free(b);
}

TEST_P(StitchPoolPolicyTest, DataAndMetadataAllocationsAreTrackedSeparately) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle data = pool.allocate(1024, MemoryKind::Data);
	StitchHandle metadata = pool.allocate(128, MemoryKind::Metadata);

	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), 1024u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Metadata), 128u);
	EXPECT_EQ(pool.bytesAllocated(), 1024u + 128u);

	pool.free(data);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), 0u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Metadata), 128u);  // untouched

	pool.free(metadata);
}

TEST_P(StitchPoolPolicyTest, MetadataAllocationRoundTripsThroughDeviceMemory) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	constexpr std::size_t kBytes = 64;
	StitchHandle handle = pool.allocate(kBytes, MemoryKind::Metadata);

	std::vector<std::byte> host_in(kBytes, std::byte{0x5a});
	pool.copyFromHost(handle, host_in.data(), kBytes);

	std::vector<std::byte> host_out(kBytes);
	pool.copyToHost(handle, host_out.data(), kBytes);
	EXPECT_EQ(host_in, host_out);

	pool.free(handle);
}

TEST_P(StitchPoolPolicyTest, CopyFromHostRejectsOversizedRequest) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(16);
	std::vector<std::byte> host_in(32);

	EXPECT_THROW(pool.copyFromHost(handle, host_in.data(), 32), std::out_of_range);

	pool.free(handle);
}

TEST_P(StitchPoolPolicyTest, AcquireReturnsLeaseWithMatchingPointer) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(128);
	void* expected = pool.access(handle, AccessMode::Read);

	{
		StitchPool::Lease lease = pool.acquire(handle);
		EXPECT_EQ(lease.ptr(), expected);
	}  // released here

	pool.free(handle);
}

TEST_P(StitchPoolPolicyTest, AcquireOfInvalidHandleThrows) {
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	EXPECT_THROW(pool.acquire(StitchHandle{}), std::invalid_argument);
}

TEST_P(StitchPoolPolicyTest, FreeWaitsForOutstandingLeaseToRelease) {
	// While a Lease is held (on any thread), free() must not reclaim the
	// memory -- it should block until the Lease is released. Proven here by
	// timing: the holder sleeps before releasing, and free() must not return
	// before that sleep elapses.
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(128);
	constexpr auto kHoldDuration = std::chrono::milliseconds(150);

	std::thread holder([&] {
		StitchPool::Lease lease = pool.acquire(handle);
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

TEST_P(StitchPoolPolicyTest, FreeWaitsForAllOutstandingLeasesAcrossDifferentStreams) {
	// Two Leases on the same handle, acquired on two different streams --
	// free() must wait for both, not just the first to release.
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(128);
	cudaStream_t stream_a = nullptr;
	cudaStream_t stream_b = nullptr;
	ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
	ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

	constexpr auto kHoldDuration = std::chrono::milliseconds(150);

	std::thread holder_a([&] {
		StitchPool::Lease lease = pool.acquire(handle, stream_a);
		std::this_thread::sleep_for(kHoldDuration);
	});
	std::thread holder_b([&] {
		StitchPool::Lease lease = pool.acquire(handle, stream_b);
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

TEST_P(StitchPoolPolicyTest, CompactRelocatesButPreservesDataAndByteTotal) {
	// compact() is invoked explicitly by the caller (Controller, as part of
	// its Eviction -> Compaction -> Promotion pipeline) whenever it's
	// already decided compaction is needed -- there's no internal occupancy
	// heuristic left to satisfy here, it always does the relocation work
	// when called (except under Naive, which has nothing to consolidate).
	DeviceContext device = MakeDevice();
	StitchPool pool(device);

	constexpr std::size_t kBytes = 128;
	StitchHandle handle = pool.allocate(kBytes, MemoryKind::Data);
	std::vector<std::byte> host_in(kBytes, std::byte{0x42});
	pool.copyFromHost(handle, host_in.data(), kBytes);

	void* before = pool.access(handle, AccessMode::Read);
	StitchPool::CompactionResult result = pool.compact(MemoryKind::Data);
	void* after = pool.access(handle, AccessMode::Read);

	if (GetParam() == AllocationPolicy::Naive) {
		EXPECT_EQ(result.relocated_count, 0u);  // no-op under Naive
		EXPECT_EQ(before, after);
	} else {
		EXPECT_EQ(result.relocated_count, 1u);
		EXPECT_EQ(result.bytes_relocated, kBytes);
	}

	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), kBytes);  // unchanged either way

	std::vector<std::byte> host_out(kBytes);
	pool.copyToHost(handle, host_out.data(), kBytes);
	EXPECT_EQ(host_in, host_out);  // data survived, wherever it now lives

	pool.free(handle);
}

TEST(StitchPoolTest, CompactWaitsForOutstandingLeaseToRelease) {
	// Same timing argument as FreeWaitsForOutstandingLeaseToRelease, but for
	// compact() -- it must not copy-and-relocate a handle out from under a
	// still-held Lease. Pooled only: compact() is a no-op under Naive
	// regardless, so there'd be nothing to time.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled);
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(128, MemoryKind::Data);
	constexpr auto kHoldDuration = std::chrono::milliseconds(150);

	std::thread holder([&] {
		StitchPool::Lease lease = pool.acquire(handle);
		std::this_thread::sleep_for(kHoldDuration);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	auto start = std::chrono::steady_clock::now();
	StitchPool::CompactionResult result = pool.compact(MemoryKind::Data);
	auto elapsed = std::chrono::steady_clock::now() - start;

	holder.join();
	EXPECT_GE(elapsed, kHoldDuration - std::chrono::milliseconds(20));
	EXPECT_EQ(result.relocated_count, 1u);

	pool.free(handle);
}

INSTANTIATE_TEST_SUITE_P(PooledAndNaive, StitchPoolPolicyTest,
													::testing::Values(AllocationPolicy::Pooled, AllocationPolicy::Naive),
													[](const ::testing::TestParamInfo<AllocationPolicy>& info) {
														return info.param == AllocationPolicy::Pooled ? "Pooled" : "Naive";
													});

TEST(StitchPoolTest, DefaultDeviceContextUsesPooledPolicy) {
	DeviceContext device;
	EXPECT_EQ(device.allocationPolicy(), AllocationPolicy::Pooled);
}

TEST(StitchPoolTest, NaivePolicyDoesNotPreReserveAnArena) {
	// Under Naive, DeviceContext shouldn't need to reserve
	// kDefaultDataPoolBytes/kDefaultMetadataPoolBytes up front -- passing
	// sizes that would be absurd to actually cudaMalloc (and would fail
	// construction under Pooled) should still succeed, since Naive ignores
	// them entirely.
	constexpr std::size_t kAbsurdlyLarge = std::size_t{1} << 40;  // 1 TiB
	EXPECT_NO_THROW(DeviceContext(/*device_id=*/0, AllocationPolicy::Naive, kAbsurdlyLarge,
																 kAbsurdlyLarge));
}

// ---------------------------------------------------------------------------
// Stress tests: many rounds of random allocate/free churn (to actually
// produce fragmentation under Pooled), then a compact() pass that must
// preserve every survivor's data and total live bytes. Sizes are kept
// small (KiB-MiB) and the simultaneously-live footprint capped well below
// the reservation so this stays fast regardless of how large that
// reservation is -- see PooledPolicySurvivesExcessiveAllocDeallocAndCompaction
// for where the actual 90%-of-free-memory reservation is exercised.
// ---------------------------------------------------------------------------

struct LiveEntry {
	StitchHandle handle;
	std::size_t bytes;
	std::byte canary;
};

// Runs `iterations` rounds of random allocate-with-data (a byte pattern
// written and later checked, to catch any corruption from the moves
// compact() performs) or free against `pool`, keeping the total
// simultaneously-live footprint under `max_live_bytes`. Returns whatever is
// still live at the end -- the caller frees it.
std::vector<LiveEntry> RunAllocDeallocChurn(StitchPool& pool, std::mt19937& rng, int iterations,
																						 std::size_t max_live_bytes) {
	std::uniform_int_distribution<std::size_t> size_dist(1024, 1024 * 1024);  // 1 KiB..1 MiB
	std::vector<LiveEntry> live;
	std::size_t live_bytes = 0;

	for (int i = 0; i < iterations; ++i) {
		bool should_allocate = live.empty() || (live_bytes < max_live_bytes && (rng() % 3 != 0));
		if (should_allocate) {
			std::size_t bytes = size_dist(rng);
			if (live_bytes + bytes > max_live_bytes) continue;

			StitchHandle handle = pool.allocate(bytes, MemoryKind::Data);
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

void VerifyLiveData(StitchPool& pool, const std::vector<LiveEntry>& live) {
	for (const LiveEntry& entry : live) {
		std::vector<std::byte> out(entry.bytes);
		pool.copyToHost(entry.handle, out.data(), entry.bytes);
		for (std::byte b : out) {
			ASSERT_EQ(b, entry.canary);
		}
	}
}

TEST(StitchPoolStressTest, PooledPolicySurvivesExcessiveAllocDeallocAndCompaction) {
	cudaSetDevice(0);
	std::size_t free_bytes = 0;
	std::size_t total_bytes = 0;
	ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);

	// The 90% figure is about proving Arachne can reserve a big-chunk arena
	// at that scale (per the "ANNS index is huge" plan), not about churning
	// gigabytes of live data through it every test run -- see
	// RunAllocDeallocChurn's kMaxLiveBytes cap below for the actual churn
	// volume.
	std::size_t budget = static_cast<std::size_t>(static_cast<double>(free_bytes) * 0.9);
	budget -= budget % 256;  // pool_memory_resource requires 256-byte alignment
	ASSERT_GT(budget, 0u);

	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, budget, kDefaultMetadataPoolBytes);
	StitchPool pool(device);

	std::mt19937 rng(12345);
	constexpr std::size_t kMaxLiveBytes = 128 * 1024 * 1024;  // 128 MiB
	constexpr int kIterations = 3000;

	std::vector<LiveEntry> live = RunAllocDeallocChurn(pool, rng, kIterations, kMaxLiveBytes);
	ASSERT_FALSE(live.empty());
	VerifyLiveData(pool, live);

	std::size_t before = pool.bytesAllocated(MemoryKind::Data);
	StitchPool::CompactionResult result = pool.compact(MemoryKind::Data);
	std::size_t after = pool.bytesAllocated(MemoryKind::Data);

	EXPECT_EQ(before, after);  // compaction changes addresses, not live totals
	EXPECT_EQ(result.relocated_count, live.size());
	EXPECT_EQ(result.bytes_relocated, before);

	VerifyLiveData(pool, live);  // data must survive relocation

	for (const LiveEntry& entry : live) pool.free(entry.handle);
}

TEST(StitchPoolStressTest, NaivePolicySurvivesExcessiveAllocDealloc) {
	// Naive still goes through RAFT/RMM's cuda_memory_resource for every
	// single allocate()/free() (real cudaMalloc/cudaFree each time, no
	// pooling) -- this exercises that path under the same kind of churn as
	// the Pooled stress test above, to catch anything that was accidentally
	// only correct when a pool was amortizing the calls.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Naive);
	StitchPool pool(device);

	std::mt19937 rng(54321);
	constexpr std::size_t kMaxLiveBytes = 64 * 1024 * 1024;  // 64 MiB
	constexpr int kIterations = 2000;

	std::vector<LiveEntry> live = RunAllocDeallocChurn(pool, rng, kIterations, kMaxLiveBytes);
	ASSERT_FALSE(live.empty());
	VerifyLiveData(pool, live);

	// compact() is a documented no-op under Naive (no shared arena to
	// consolidate) -- confirm that explicitly here rather than only relying
	// on the parametrized CompactRelocatesButPreservesDataAndByteTotal case.
	std::size_t before = pool.bytesAllocated(MemoryKind::Data);
	StitchPool::CompactionResult result = pool.compact(MemoryKind::Data);

	EXPECT_EQ(result.relocated_count, 0u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), before);
	VerifyLiveData(pool, live);  // untouched, as expected

	for (const LiveEntry& entry : live) pool.free(entry.handle);
}

}  // namespace
