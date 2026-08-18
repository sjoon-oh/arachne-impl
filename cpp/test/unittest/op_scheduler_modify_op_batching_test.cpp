// End-to-end (real OpScheduler, not just FifoSchedulingPolicy in isolation)
// check of the op-homogeneity invariant added to SchedulingPolicy::
// canAppendToBatch() (see scheduling_policy.hpp's class doc comment):
// concurrently submits a mix of Insert and Delete ModifyRequests with
// modify_batch_size configured well above 1, and asserts the adapter's
// modifyHost() never once receives a batch mixing both ops -- op_scheduler_
// policy_test.cpp already covers the policy decision in isolation; this
// covers the same invariant actually holding under real concurrent
// scheduling.

#include "core/op_scheduler.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "adapter/index_adapter.hpp"

namespace {

using namespace arachne;

class RecordingAdapter : public IAdapter {
 public:
	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		return std::vector<TraverseResult>(requests.size());
	}

	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			batch_sizes_.push_back(requests.size());
			bool all_insert = true;
			bool all_delete = true;
			for (const ModifyRequest& request : requests) {
				if (request.op != ModifyOp::Insert) all_insert = false;
				if (request.op != ModifyOp::Delete) all_delete = false;
			}
			if (!all_insert && !all_delete) saw_mixed_batch_ = true;
		}
		// Small delay so concurrently-submitted requests have a real chance to
		// pile up in the scheduler's queue while this call is in flight,
		// instead of always draining one at a time before the next submission
		// even reaches the queue.
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
		std::vector<ModifyResult> results(requests.size());
		for (ModifyResult& result : results) result.ok = true;
		return results;
	}

	IRegion* resolveRegion(RegionId) override { return nullptr; }
	std::vector<RegionId> allRegions() const override { return {}; }

	bool sawMixedBatch() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return saw_mixed_batch_;
	}
	std::size_t maxBatchSize() const {
		std::lock_guard<std::mutex> lock(mutex_);
		std::size_t max_size = 0;
		for (std::size_t size : batch_sizes_) max_size = std::max(max_size, size);
		return max_size;
	}
	std::size_t batchCount() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return batch_sizes_.size();
	}

 private:
	mutable std::mutex mutex_;
	std::vector<std::size_t> batch_sizes_;
	bool saw_mixed_batch_ = false;
};

TEST(OpSchedulerModifyOpBatchingTest, NeverPassesAMixedOpBatchToTheAdapter) {
	constexpr std::size_t kSubmitterThreads = 6;
	constexpr std::size_t kRequestsPerThread = 40;

	SchedulingConfig config;
	config.modify_batch_size = 16;  // well above 1, so batching actually has room to happen
	config.max_execution_threads = 3;
	config.batch_wait_timeout = std::chrono::microseconds(2000);  // small grace window to let concurrent submits join

	RecordingAdapter adapter;
	OpScheduler scheduler(config);
	scheduler.start(adapter);

	std::vector<std::thread> submitters;
	std::mutex futures_mutex;
	std::vector<std::future<ModifyResult>> futures;

	for (std::size_t t = 0; t < kSubmitterThreads; ++t) {
		submitters.emplace_back([&, t] {
			for (std::size_t i = 0; i < kRequestsPerThread; ++i) {
				ModifyRequest request;
				// Alternate Insert/Delete both across threads and within a thread's
				// own sequence, so the queue genuinely interleaves both ops rather
				// than happening to arrive in same-op clumps.
				request.op = ((t + i) % 2 == 0) ? ModifyOp::Insert : ModifyOp::Delete;
				request.target = static_cast<VectorId>(t * kRequestsPerThread + i) + 1;
				std::future<ModifyResult> future = scheduler.schedule(request);
				std::lock_guard<std::mutex> lock(futures_mutex);
				futures.push_back(std::move(future));
			}
		});
	}
	for (auto& thread : submitters) thread.join();

	for (auto& future : futures) {
		ModifyResult result = future.get();
		EXPECT_TRUE(result.ok);
	}
	scheduler.shutdown();

	EXPECT_FALSE(adapter.sawMixedBatch()) << "op-homogeneity invariant violated -- a batch mixed Insert and Delete";
	EXPECT_GT(adapter.batchCount(), 0u);
	EXPECT_GT(adapter.maxBatchSize(), 1u)
			<< "every batch was size 1 -- this test isn't actually exercising batching, only isolation";
}

}  // namespace
