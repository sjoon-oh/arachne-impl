#include "gpu/device_context.hpp"

#include <cuda_runtime.h>

#include <rmm/mr/pool_memory_resource.hpp>
#include <stdexcept>
#include <string>

#include "logging.hpp"

namespace arachne::gpu {

namespace {

int SetActiveDevice(int device_id) {
	cudaError_t status = cudaSetDevice(device_id);
	if (status != cudaSuccess) {
		throw std::runtime_error("DeviceContext: cudaSetDevice(" + std::to_string(device_id) +
															") failed: " + cudaGetErrorString(status));
	}
	return device_id;
}

// Type-erases either strategy behind the same any_resource<device_accessible>
// DeviceRegionPool calls allocate()/deallocate() on -- see AllocationPolicy's doc
// comment. `upstream` is copied into whichever concrete resource gets
// built (cheap: cuda_memory_resource is stateless, "all instances
// equivalent").
cuda::mr::any_resource<cuda::mr::device_accessible> MakeResource(AllocationPolicy policy,
																																	 rmm::mr::cuda_memory_resource upstream,
																																	 std::size_t initial_pool_bytes) {
	switch (policy) {
		case AllocationPolicy::Naive:
			return cuda::mr::any_resource<cuda::mr::device_accessible>{upstream};
		case AllocationPolicy::Pooled:
		default:
			return cuda::mr::any_resource<cuda::mr::device_accessible>{
					rmm::mr::pool_memory_resource{upstream, initial_pool_bytes}};
	}
}

const char* ToString(AllocationPolicy policy) {
	switch (policy) {
		case AllocationPolicy::Pooled:
			return "Pooled";
		case AllocationPolicy::Naive:
			return "Naive";
	}
	return "Unknown";
}

}  // namespace

DeviceContext::DeviceContext(int device_id, AllocationPolicy policy, std::size_t data_pool_bytes,
															std::size_t metadata_pool_bytes)
		: device_id_(SetActiveDevice(device_id)),
			policy_(policy),
			data_pool_bytes_(data_pool_bytes),
			metadata_pool_bytes_(metadata_pool_bytes),
			data_resource_(MakeResource(policy_, memory_resource_, data_pool_bytes)),
			metadata_resource_(MakeResource(policy_, memory_resource_, metadata_pool_bytes)) {
	ARACHNE_LOG_INFO(
			"DeviceContext: initialized CUDA device {} (policy={}, data pool {} bytes, metadata pool "
			"{} bytes)",
			device_id_, ToString(policy_), data_pool_bytes, metadata_pool_bytes);
}

}  // namespace arachne::gpu
