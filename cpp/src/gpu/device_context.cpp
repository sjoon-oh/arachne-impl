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
		case AllocationPolicy::Async:
			return "Async";
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
	if (worker_scratch_base_ != nullptr) cudaFree(worker_scratch_base_);
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

void* DeviceContext::workerScratch(std::size_t index) const {
	if (worker_scratch_base_ == nullptr) return nullptr;  // reserveWorkerScratch() never called, or bytes_per_worker==0
	if (index >= worker_streams_.size()) {
		throw std::out_of_range("DeviceContext::workerScratch: index out of range");
	}
	return static_cast<char*>(worker_scratch_base_) + index * worker_scratch_bytes_;
}

void DeviceContext::reserveWorkerScratch(std::size_t bytes_per_worker) {
	if (worker_scratch_bytes_ != 0 || worker_scratch_base_ != nullptr) {
		throw std::logic_error("DeviceContext::reserveWorkerScratch: already reserved");
	}
	worker_scratch_bytes_ = bytes_per_worker;
	if (bytes_per_worker == 0) return;  // valid no-op -- workerScratch() keeps returning nullptr

	std::size_t total_bytes = bytes_per_worker * worker_streams_.size();
	cudaError_t status = cudaMalloc(&worker_scratch_base_, total_bytes);
	if (status != cudaSuccess) {
		throw std::runtime_error("DeviceContext::reserveWorkerScratch: cudaMalloc(" + std::to_string(total_bytes) +
															" bytes) failed: " + cudaGetErrorString(status));
	}
	ARACHNE_LOG_INFO("DeviceContext::reserveWorkerScratch: reserved {} bytes/worker ({} bytes total, {} worker(s))",
										bytes_per_worker, total_bytes, worker_streams_.size());
}

}  // namespace arachne::gpu
