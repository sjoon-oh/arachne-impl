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

const char* ToString(AllocationPolicy policy) {
	switch (policy) {
		case AllocationPolicy::Pooled:
			return "Pooled";
		case AllocationPolicy::Normal:
			return "Normal";
	}
	return "Unknown";
}

}  // namespace

DeviceContext::DeviceContext(int device_id, AllocationPolicy policy, std::size_t data_pool_bytes,
															std::size_t metadata_pool_bytes, std::size_t worker_stream_count,
															std::size_t unit_bytes)
		: device_id_(SetActiveDevice(device_id)),
			policy_(policy),
			data_pool_bytes_(data_pool_bytes),
			metadata_pool_bytes_(metadata_pool_bytes),
			// Always constructed, even though Pooled leaves these two unused (see
			// dataResource()/metadataResource()'s own doc comment) -- keeps those
			// accessors valid regardless of `policy`.
			data_resource_(memory_resource_),
			metadata_resource_(memory_resource_) {
	worker_stream_count = worker_stream_count == 0 ? 1 : worker_stream_count;
	worker_streams_.reserve(worker_stream_count);
	for (std::size_t i = 0; i < worker_stream_count; ++i) {
		cudaStream_t stream = nullptr;
		cudaError_t status = cudaStreamCreate(&stream);
		if (status != cudaSuccess) {
			throw std::runtime_error(std::string("DeviceContext: cudaStreamCreate failed: ") +
																cudaGetErrorString(status));
		}
		worker_streams_.push_back(stream);
	}

	if (policy_ == AllocationPolicy::Pooled) {
		data_arena_ = std::make_unique<UnitPoolArena>(data_resource_, resources_, unit_bytes, data_pool_bytes);
		metadata_arena_ =
				std::make_unique<UnitPoolArena>(metadata_resource_, resources_, unit_bytes, metadata_pool_bytes);
	}

	ARACHNE_LOG_INFO(
			"DeviceContext: initialized CUDA device {} (policy={}, data pool {} bytes, metadata pool "
			"{} bytes, {} worker stream(s))",
			device_id_, ToString(policy_), data_pool_bytes, metadata_pool_bytes, worker_stream_count);
}

DeviceContext::~DeviceContext() {
	for (cudaStream_t stream : worker_streams_) cudaStreamDestroy(stream);
}

std::size_t DeviceContext::budgetBytes(MemoryKind kind) const {
	if (policy_ == AllocationPolicy::Pooled) {
		const UnitPoolArena& arena = (kind == MemoryKind::Metadata) ? *metadata_arena_ : *data_arena_;
		return arena.totalUnits() * arena.unitBytes();
	}
	return kind == MemoryKind::Metadata ? metadata_pool_bytes_ : data_pool_bytes_;
}

cudaStream_t DeviceContext::workerStream(std::size_t index) const {
	if (index >= worker_streams_.size()) {
		throw std::out_of_range("DeviceContext::workerStream: index out of range");
	}
	return worker_streams_[index];
}

}  // namespace arachne::gpu
