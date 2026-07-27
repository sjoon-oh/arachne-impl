#include "gpu/stitch_pool.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "gpu/device_context.hpp"

namespace {

using arachne::gpu::AccessMode;
using arachne::gpu::DeviceContext;
using arachne::gpu::MemoryKind;
using arachne::gpu::StitchHandle;
using arachne::gpu::StitchPool;

TEST(StitchPoolTest, AllocateReturnsValidHandleAndTracksBytes) {
	DeviceContext device;
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(1024);

	EXPECT_TRUE(handle.valid());
	EXPECT_EQ(pool.bytesAllocated(), 1024u);
}

TEST(StitchPoolTest, AccessReturnsDereferenceableDeviceMemory) {
	DeviceContext device;
	StitchPool pool(device);

	constexpr std::size_t kBytes = 256;
	StitchHandle handle = pool.allocate(kBytes);
	void* device_ptr = pool.access(handle, AccessMode::Write);
	ASSERT_NE(device_ptr, nullptr);

	std::vector<std::byte> host_in(kBytes);
	for (std::size_t i = 0; i < kBytes; ++i) host_in[i] = std::byte{static_cast<unsigned char>(i)};

	ASSERT_EQ(cudaMemcpy(device_ptr, host_in.data(), kBytes, cudaMemcpyHostToDevice), cudaSuccess);

	std::vector<std::byte> host_out(kBytes);
	void* read_ptr = pool.access(handle, AccessMode::Read);
	ASSERT_EQ(cudaMemcpy(host_out.data(), read_ptr, kBytes, cudaMemcpyDeviceToHost), cudaSuccess);

	EXPECT_EQ(host_in, host_out);

	pool.free(handle);
}

TEST(StitchPoolTest, FreeReclaimsBytesAndInvalidatesHandle) {
	DeviceContext device;
	StitchPool pool(device);

	StitchHandle handle = pool.allocate(512);
	ASSERT_EQ(pool.bytesAllocated(), 512u);

	pool.free(handle);

	EXPECT_EQ(pool.bytesAllocated(), 0u);
	EXPECT_THROW(pool.access(handle, AccessMode::Read), std::invalid_argument);
}

TEST(StitchPoolTest, FreeOfInvalidHandleIsANoop) {
	DeviceContext device;
	StitchPool pool(device);

	pool.free(StitchHandle{});  // default-constructed, never allocated

	EXPECT_EQ(pool.bytesAllocated(), 0u);
}

TEST(StitchPoolTest, AccessOfNeverAllocatedHandleThrows) {
	DeviceContext device;
	StitchPool pool(device);

	EXPECT_THROW(pool.access(StitchHandle{}, AccessMode::Read), std::invalid_argument);
}

TEST(StitchPoolTest, MultipleAllocationsAreIndependent) {
	DeviceContext device;
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

TEST(StitchPoolTest, DataAndMetadataAllocationsAreTrackedSeparately) {
	DeviceContext device;
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

TEST(StitchPoolTest, MetadataAllocationIsAccessibleDeviceMemory) {
	DeviceContext device;
	StitchPool pool(device);

	constexpr std::size_t kBytes = 64;
	StitchHandle handle = pool.allocate(kBytes, MemoryKind::Metadata);
	void* device_ptr = pool.access(handle, AccessMode::Write);
	ASSERT_NE(device_ptr, nullptr);

	std::vector<std::byte> host_in(kBytes, std::byte{0x5a});
	ASSERT_EQ(cudaMemcpy(device_ptr, host_in.data(), kBytes, cudaMemcpyHostToDevice), cudaSuccess);

	std::vector<std::byte> host_out(kBytes);
	ASSERT_EQ(cudaMemcpy(host_out.data(), device_ptr, kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
	EXPECT_EQ(host_in, host_out);

	pool.free(handle);
}

}  // namespace
