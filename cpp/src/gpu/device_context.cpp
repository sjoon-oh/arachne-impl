#include "gpu/device_context.hpp"

#include <cuda_runtime.h>

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

}  // namespace

DeviceContext::DeviceContext(int device_id, std::size_t data_pool_bytes,
															std::size_t metadata_pool_bytes)
		: device_id_(SetActiveDevice(device_id)),
			data_pool_(memory_resource_, data_pool_bytes),
			metadata_pool_(memory_resource_, metadata_pool_bytes) {
	ARACHNE_LOG_INFO(
			"DeviceContext: initialized CUDA device {} (data pool {} bytes, metadata pool {} bytes)",
			device_id_, data_pool_bytes, metadata_pool_bytes);
}

}  // namespace arachne::gpu
