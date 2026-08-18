// Verifies the Traverse/Modify execution-admission gate added to OpScheduler
// (see op_scheduler.hpp's class doc comment and IAdapter::
// requiresTraverseModifyIsolation()): concurrently submits a mix of Traverse
// and Insert/Delete Modify requests against a real OpScheduler with several
// worker threads, and inspects the wall-clock [start,end) interval of every
// adapter call to prove the exact admission rule end-to-end. This is
// deliberately not just "mismatched batches are rejected" (op_scheduler_
// policy_test.cpp already covers SchedulingPolicy's own, narrower,
// within-batch invariant) -- it proves the *cross-batch* rule actually holds
// under real concurrent scheduling, and just as importantly that it doesn't
// over-serialize: same-kind-compatible batches really do overlap in
// wall-clock time, and an adapter that opts out via
// requiresTraverseModifyIsolation() == false really does see overlaps the
// default would have forbidden.

#include "core/op_scheduler.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "adapter/index_adapter.hpp"

namespace {

using namespace arachne;
using Clock = std::chrono::steady_clock;

struct CallInterval {
	ScheduledKind kind;
	std::optional<ModifyOp> op;  // nullopt for Traverse
	Clock::time_point start;
	Clock::time_point end;
};

bool Overlaps(const CallInterval& a, const CallInterval& b) { return a.start < b.end && b.start < a.end; }

// Records the wall-clock interval of every traverseHost()/modifyHost() call,
// holding each call open for `hold` so concurrently-dispatched batches have a
// real chance to overlap in time instead of racing through too fast to ever
// measurably coincide.
class IntervalRecordingAdapter : public IAdapter {
 public:
	IntervalRecordingAdapter(bool isolation, std::chrono::milliseconds hold) : isolation_(isolation), hold_(hold) {}

	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		record(ScheduledKind::Traverse, std::nullopt);
		return std::vector<TraverseResult>(requests.size());
	}
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		record(ScheduledKind::Modify, requests.front().op);
		std::vector<ModifyResult> results(requests.size());
		for (ModifyResult& result : results) result.ok = true;
		return results;
	}
	IRegion* resolveRegion(RegionId) override { return nullptr; }
	std::vector<RegionId> allRegions() const override { return {}; }
	bool requiresTraverseModifyIsolation() const override { return isolation_; }

	std::vector<CallInterval> intervals() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return intervals_;
	}

 private:
	void record(ScheduledKind kind, std::optional<ModifyOp> op) {
		Clock::time_point start = Clock::now();
		std::this_thread::sleep_for(hold_);
		Clock::time_point end = Clock::now();
		std::lock_guard<std::mutex> lock(mutex_);
		intervals_.push_back({kind, op, start, end});
	}

	bool isolation_;
	std::chrono::milliseconds hold_;
	mutable std::mutex mutex_;
	std::vector<CallInterval> intervals_;
};

// Fires a mix of Traverse, Insert and Delete requests at `scheduler` from
// several concurrent submitter threads (so the planner sees a genuinely
// interleaved queue rather than same-kind runs), and blocks until every
// request's result has been observed.
void SubmitMixedWorkloadAndWait(OpScheduler& scheduler, std::size_t requests_per_thread, std::size_t threads_per_kind) {
	std::mutex futures_mutex;
	std::vector<std::future<TraverseResult>> traverse_futures;
	std::vector<std::future<ModifyResult>> modify_futures;
	std::vector<std::thread> submitters;

	auto submit_traverse = [&] {
		for (std::size_t i = 0; i < requests_per_thread; ++i) {
			auto future = scheduler.schedule(TraverseRequest{});
			std::lock_guard<std::mutex> lock(futures_mutex);
			traverse_futures.push_back(std::move(future));
		}
	};
	auto submit_modify = [&](ModifyOp op, std::uint64_t base) {
		for (std::size_t i = 0; i < requests_per_thread; ++i) {
			ModifyRequest request;
			request.op = op;
			request.target = static_cast<VectorId>(base + i) + 1;
			auto future = scheduler.schedule(request);
			std::lock_guard<std::mutex> lock(futures_mutex);
			modify_futures.push_back(std::move(future));
		}
	};

	for (std::size_t t = 0; t < threads_per_kind; ++t) {
		submitters.emplace_back(submit_traverse);
		submitters.emplace_back(submit_modify, ModifyOp::Insert, t * 1000);
		submitters.emplace_back(submit_modify, ModifyOp::Delete, t * 1000 + 500);
	}
	for (auto& thread : submitters) thread.join();

	for (auto& future : traverse_futures) future.get();
	for (auto& future : modify_futures) future.get();
}

TEST(OpSchedulerTraverseModifyIsolationTest, DefaultIsolationSeparatesTraverseAndCrossOpModifyButAllowsRest) {
	IntervalRecordingAdapter adapter(/*isolation=*/true, std::chrono::milliseconds(20));

	SchedulingConfig config;
	config.traverse_batch_size = 1;
	config.modify_batch_size = 1;
	config.max_execution_threads = 6;

	OpScheduler scheduler(config);
	scheduler.start(adapter);
	SubmitMixedWorkloadAndWait(scheduler, /*requests_per_thread=*/4, /*threads_per_kind=*/3);
	scheduler.shutdown();

	std::vector<CallInterval> intervals = adapter.intervals();
	ASSERT_GT(intervals.size(), 0u);

	bool traverse_traverse_overlap = false;
	bool same_op_modify_overlap = false;
	for (std::size_t i = 0; i < intervals.size(); ++i) {
		for (std::size_t j = i + 1; j < intervals.size(); ++j) {
			const CallInterval& a = intervals[i];
			const CallInterval& b = intervals[j];
			if (!Overlaps(a, b)) continue;

			if (a.kind == ScheduledKind::Traverse && b.kind == ScheduledKind::Traverse) {
				traverse_traverse_overlap = true;
				continue;
			}
			if (a.kind == ScheduledKind::Modify && b.kind == ScheduledKind::Modify && a.op == b.op) {
				same_op_modify_overlap = true;
				continue;
			}

			// Anything else overlapping is exactly what the gate must forbid:
			// Traverse-vs-Modify, or Modify-vs-Modify with different ops.
			ADD_FAILURE() << "gate violated -- overlapping calls that must be mutually exclusive: a.kind="
										<< static_cast<int>(a.kind) << " a.op=" << (a.op ? static_cast<int>(*a.op) : -1)
										<< " vs b.kind=" << static_cast<int>(b.kind) << " b.op=" << (b.op ? static_cast<int>(*b.op) : -1);
		}
	}

	EXPECT_TRUE(traverse_traverse_overlap)
			<< "Traverse batches never overlapped -- this test isn't actually exercising concurrency, only isolation";
	EXPECT_TRUE(same_op_modify_overlap) << "same-op Modify batches never overlapped -- this test isn't actually "
																					"exercising concurrency, only isolation";
}

TEST(OpSchedulerTraverseModifyIsolationTest, OptedOutAdapterSeesNoGateAtAll) {
	IntervalRecordingAdapter adapter(/*isolation=*/false, std::chrono::milliseconds(20));

	SchedulingConfig config;
	config.traverse_batch_size = 1;
	config.modify_batch_size = 1;
	config.max_execution_threads = 6;

	OpScheduler scheduler(config);
	scheduler.start(adapter);
	SubmitMixedWorkloadAndWait(scheduler, /*requests_per_thread=*/4, /*threads_per_kind=*/3);
	scheduler.shutdown();

	std::vector<CallInterval> intervals = adapter.intervals();
	ASSERT_GT(intervals.size(), 0u);

	bool traverse_modify_overlap = false;
	for (std::size_t i = 0; i < intervals.size() && !traverse_modify_overlap; ++i) {
		for (std::size_t j = i + 1; j < intervals.size() && !traverse_modify_overlap; ++j) {
			const CallInterval& a = intervals[i];
			const CallInterval& b = intervals[j];
			if (a.kind != b.kind && Overlaps(a, b)) traverse_modify_overlap = true;
		}
	}

	EXPECT_TRUE(traverse_modify_overlap)
			<< "no Traverse/Modify overlap ever observed -- requiresTraverseModifyIsolation()==false should mean the "
				 "gate never blocks anything, same as the feature not existing";
}

}  // namespace
