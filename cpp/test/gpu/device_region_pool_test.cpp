#include "gpu/device_region_pool.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpu/device_context.hpp"

namespace {

using arachne::gpu::AllocationPolicy;
using arachne::gpu::DeviceContext;
using arachne::gpu::kDefaultDataPoolBytes;
using arachne::gpu::kDefaultMetadataPoolBytes;
using arachne::gpu::MemoryKind;
using arachne::gpu::DeviceRegionHandle;
using arachne::gpu::DeviceRegionPool;

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
	// Three independent allocations, each given a distinct byte pattern,
	// read back via three enqueueCopyToHost() calls sharing one `pending`
	// and a single flush() -- proves the batched (Controller::
	// writeBackDirtyRegions()'s) gather path actually lands every copy
	// correctly, not just the single-handle copyToHost() convenience path.
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
	// Budget deliberately tiny so a modest request already exceeds it,
	// regardless of how much real GPU memory is actually free.
	DeviceContext device(/*device_id=*/0, GetParam(), /*data_pool_bytes=*/1024,
											 arachne::gpu::kDefaultMetadataPoolBytes);
	DeviceRegionPool pool(device);

	std::optional<DeviceRegionHandle> handle = pool.tryAllocate(2048);
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
		// A second, separate acquire() (no compact() in between) must resolve
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

TEST_P(DeviceRegionPoolPolicyTest, CompactRelocatesButPreservesDataAndByteTotal) {
	// compact() is invoked explicitly by the caller (Controller, as part of
	// its Eviction -> Compaction -> Promotion pipeline) whenever it's
	// already decided compaction is needed -- there's no internal occupancy
	// heuristic left to satisfy here, it always does the relocation work
	// when called (except under Naive, which has nothing to consolidate).
	DeviceContext device = MakeDevice();
	DeviceRegionPool pool(device);

	constexpr std::size_t kBytes = 128;
	DeviceRegionHandle handle = pool.allocate(kBytes, MemoryKind::Data);
	std::vector<std::byte> host_in(kBytes, std::byte{0x42});
	pool.copyFromHost(handle, host_in.data(), kBytes);

	void* before;
	{
		DeviceRegionPool::Lease lease = pool.acquire(handle);
		before = lease.ptr();
	}
	DeviceRegionPool::CompactionResult result = pool.compact(MemoryKind::Data);
	void* after;
	{
		DeviceRegionPool::Lease lease = pool.acquire(handle);
		after = lease.ptr();
	}

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

TEST(DeviceRegionPoolTest, CompactWaitsForOutstandingLeaseToRelease) {
	// Same timing argument as FreeWaitsForOutstandingLeaseToRelease, but for
	// compact() -- it must not copy-and-relocate a handle out from under a
	// still-held Lease. Pooled only: compact() is a no-op under Naive
	// regardless, so there'd be nothing to time.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled);
	DeviceRegionPool pool(device);

	DeviceRegionHandle handle = pool.allocate(128, MemoryKind::Data);
	constexpr auto kHoldDuration = std::chrono::milliseconds(150);

	std::thread holder([&] {
		DeviceRegionPool::Lease lease = pool.acquire(handle);
		std::this_thread::sleep_for(kHoldDuration);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	auto start = std::chrono::steady_clock::now();
	DeviceRegionPool::CompactionResult result = pool.compact(MemoryKind::Data);
	auto elapsed = std::chrono::steady_clock::now() - start;

	holder.join();
	EXPECT_GE(elapsed, kHoldDuration - std::chrono::milliseconds(20));
	EXPECT_EQ(result.relocated_count, 1u);

	pool.free(handle);
}

TEST_P(DeviceRegionPoolPolicyTest, CrossStreamAcquireOrdersAfterPriorStreamsReleasedWork) {
	// Exercises acquire()'s cross-stream event-wait (see its own doc comment):
	// a Lease released on stream A, with its enqueued write not yet known to
	// have landed, followed immediately by a *different* stream B acquiring
	// the same handle -- with no host-side sync in between. If acquire()
	// didn't insert a GPU-side wait for stream A's outstanding work before
	// handing back the stream B Lease, stream B's read below could run before
	// stream A's write actually completes, reading stale/undefined bytes
	// instead of `pattern`. Large enough (1 MiB) that the write is not
	// instantaneous, to give a real window for this race to manifest if the
	// wait were missing.
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

INSTANTIATE_TEST_SUITE_P(PooledAndNaive, DeviceRegionPoolPolicyTest,
													::testing::Values(AllocationPolicy::Pooled, AllocationPolicy::Naive),
													[](const ::testing::TestParamInfo<AllocationPolicy>& info) {
														return info.param == AllocationPolicy::Pooled ? "Pooled" : "Naive";
													});

TEST(DeviceRegionPoolTest, DefaultDeviceContextUsesPooledPolicy) {
	DeviceContext device;
	EXPECT_EQ(device.allocationPolicy(), AllocationPolicy::Pooled);
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
	// Controller relies on this to route promotion/eviction traffic (the
	// management stream) and per-worker kernel launches (the worker streams)
	// onto genuinely independent CUDA streams -- see
	// DeviceContext::managementStream()/workerStream()'s own doc comments for
	// why that separation matters.
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

TEST(DeviceRegionPoolTest, NaivePolicyDoesNotPreReserveAnArena) {
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
	DeviceRegionHandle handle;
	std::size_t bytes;
	std::byte canary;
};

// Runs `iterations` rounds of random allocate-with-data (a byte pattern
// written and later checked, to catch any corruption from the moves
// compact() performs) or free against `pool`, keeping the total
// simultaneously-live footprint under `max_live_bytes`. Returns whatever is
// still live at the end -- the caller frees it.
std::vector<LiveEntry> RunAllocDeallocChurn(DeviceRegionPool& pool, std::mt19937& rng, int iterations,
																						 std::size_t max_live_bytes) {
	std::uniform_int_distribution<std::size_t> size_dist(1024, 1024 * 1024);  // 1 KiB..1 MiB
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

	// The 90% figure is about proving Arachne can reserve a big-chunk arena
	// at that scale (per the "ANNS index is huge" plan), not about churning
	// gigabytes of live data through it every test run -- see
	// RunAllocDeallocChurn's kMaxLiveBytes cap below for the actual churn
	// volume.
	std::size_t budget = static_cast<std::size_t>(static_cast<double>(free_bytes) * 0.9);
	budget -= budget % 256;  // pool_memory_resource requires 256-byte alignment
	ASSERT_GT(budget, 0u);

	DeviceContext device(/*device_id=*/0, AllocationPolicy::Pooled, budget, kDefaultMetadataPoolBytes);
	DeviceRegionPool pool(device);

	std::mt19937 rng(12345);
	constexpr std::size_t kMaxLiveBytes = 128 * 1024 * 1024;  // 128 MiB
	constexpr int kIterations = 3000;

	std::vector<LiveEntry> live = RunAllocDeallocChurn(pool, rng, kIterations, kMaxLiveBytes);
	ASSERT_FALSE(live.empty());
	VerifyLiveData(pool, live);

	std::size_t before = pool.bytesAllocated(MemoryKind::Data);
	DeviceRegionPool::CompactionResult result = pool.compact(MemoryKind::Data);
	std::size_t after = pool.bytesAllocated(MemoryKind::Data);

	EXPECT_EQ(before, after);  // compaction changes addresses, not live totals
	EXPECT_EQ(result.relocated_count, live.size());
	EXPECT_EQ(result.bytes_relocated, before);

	VerifyLiveData(pool, live);  // data must survive relocation

	for (const LiveEntry& entry : live) pool.free(entry.handle);
}

TEST(DeviceRegionPoolStressTest, NaivePolicySurvivesExcessiveAllocDealloc) {
	// Naive still goes through RAFT/RMM's cuda_memory_resource for every
	// single allocate()/free() (real cudaMalloc/cudaFree each time, no
	// pooling) -- this exercises that path under the same kind of churn as
	// the Pooled stress test above, to catch anything that was accidentally
	// only correct when a pool was amortizing the calls.
	DeviceContext device(/*device_id=*/0, AllocationPolicy::Naive);
	DeviceRegionPool pool(device);

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
	DeviceRegionPool::CompactionResult result = pool.compact(MemoryKind::Data);

	EXPECT_EQ(result.relocated_count, 0u);
	EXPECT_EQ(pool.bytesAllocated(MemoryKind::Data), before);
	VerifyLiveData(pool, live);  // untouched, as expected

	for (const LiveEntry& entry : live) pool.free(entry.handle);
}

}  // namespace
