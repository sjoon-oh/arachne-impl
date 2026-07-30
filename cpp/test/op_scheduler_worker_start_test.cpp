// Tests OpScheduler's on_worker_start hook (see op_scheduler.hpp's start()
// doc comment): Controller uses this to bind each execution worker thread
// to its own dedicated CUDA stream via thread-local state, without
// OpScheduler needing to know CUDA exists. These tests stay GPU-agnostic --
// they only verify the hook itself fires correctly (once per worker,
// distinct 0-based indices), which is all OpScheduler is responsible for.

#include "core/op_scheduler.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "adapter/index_adapter.hpp"

namespace {

using namespace arachne;

class FakeAdapter : public IAdapter {
 public:
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		return std::vector<TraverseResult>(requests.size());
	}
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		std::vector<ModifyResult> results(requests.size());
		for (ModifyResult& result : results) result.ok = true;
		return results;
	}
	IRegion* resolveRegion(RegionId) override { return nullptr; }
	std::vector<RegionId> allRegions() const override { return {}; }
};

TEST(OpSchedulerWorkerStartTest, OnWorkerStartCalledOnceForEachWorkerWithDistinctIndices) {
	constexpr std::size_t kWorkers = 4;
	SchedulingConfig config;
	config.max_execution_threads = kWorkers;
	OpScheduler scheduler(config);
	FakeAdapter adapter;

	std::mutex mutex;
	std::vector<std::size_t> seen_indices;  // one push per call -- also catches a double-call bug, not just a set

	scheduler.start(adapter, [&](std::size_t worker_index) {
		std::lock_guard<std::mutex> lock(mutex);
		seen_indices.push_back(worker_index);
	});

	// Poll (bounded) rather than a flat sleep: workers should register within
	// milliseconds of start(), but nothing guarantees exactly when each OS
	// thread actually gets scheduled.
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
	while (std::chrono::steady_clock::now() < deadline) {
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (seen_indices.size() >= kWorkers) break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	scheduler.shutdown();

	std::lock_guard<std::mutex> lock(mutex);
	ASSERT_EQ(seen_indices.size(), kWorkers) << "on_worker_start should fire exactly once per worker";
	std::set<std::size_t> unique_indices(seen_indices.begin(), seen_indices.end());
	ASSERT_EQ(unique_indices.size(), kWorkers) << "every worker should report a distinct index";
	for (std::size_t i = 0; i < kWorkers; ++i) {
		EXPECT_TRUE(unique_indices.count(i)) << "worker index " << i << " never called on_worker_start";
	}
}

TEST(OpSchedulerWorkerStartTest, StartWithoutCallbackStillWorks) {
	// The on_worker_start parameter is optional (defaults to nullptr) -- the
	// pre-existing one-argument start(adapter) call shape must keep working
	// unchanged for callers that don't care about worker/stream binding.
	OpScheduler scheduler;
	FakeAdapter adapter;
	ASSERT_NO_THROW(scheduler.start(adapter));

	ModifyRequest request;
	request.op = ModifyOp::Insert;
	std::future<ModifyResult> future = scheduler.schedule(request);
	ModifyResult result = future.get();
	EXPECT_TRUE(result.ok);

	scheduler.shutdown();
}

}  // namespace
