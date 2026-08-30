// Verifies two properties of OpScheduler::plannerLoop()'s batch-kind
// selection (op_scheduler.hpp's class doc comment, "Choosing *which*
// (kind, op) class..." and "Starvation override" paragraphs), neither of
// which op_scheduler_traverse_modify_isolation_test.cpp exercises (that file
// proves the admission *rule* itself; this file proves the planner doesn't
// gratuitously block when the rule would actually allow other pending work
// to proceed, and that a pluggable SchedulingPolicy that never revisits a
// given (kind, op) can't starve it forever):
//
//   1. DoesNotBlockOnAnInadmissibleChoiceWhenAnAdmissibleAlternativeExists:
//      with the default FifoSchedulingPolicy, queues a Delete in front of an
//      Insert while a first Insert is already executing. Before this
//      session's change, the planner would commit to (Modify, Delete) --
//      queue_.front()'s op -- and block until the in-flight Insert finished,
//      even though the queued Insert was safe to run concurrently with it
//      the whole time. Now it should run both Inserts concurrently instead.
//   2 & 3. StarvationOverrideDisabled/Enabled: a SchedulingPolicy that
//      always prefers Traverse whenever any is pending would, without the
//      starvation override, keep re-picking Traverse over an equally
//      admissible but older Insert purely out of static bias -- not because
//      Insert is actually blocked. Constructs one deterministic decision
//      point where both are genuinely, simultaneously admissible (see
//      OrderRecordingAdapter's own comment for why that needs a
//      synchronization gate rather than just racing real time), and proves
//      the disabled (default) case respects the policy's preference while
//      the enabled case overrides it once Insert has waited past
//      SchedulingConfig::starvation_threshold -- a property FifoScheduling
//      Policy itself never needs (see op_scheduler.hpp's own doc comment).

#include "core/op_scheduler.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "core/scheduling_policy.hpp"

namespace {

using namespace arachne;
using Clock = std::chrono::steady_clock;

struct CallInterval {
	ScheduledKind kind;
	std::optional<ModifyOp> op;
	Clock::time_point start;
	Clock::time_point end;
};

bool Overlaps(const CallInterval& a, const CallInterval& b) { return a.start < b.end && b.start < a.end; }

// Records the wall-clock interval of every traverseHost()/modifyHost() call
// (holding each open for `hold` so concurrent batches have a real chance to
// measurably overlap, same convention as op_scheduler_traverse_modify_
// isolation_test.cpp's own IntervalRecordingAdapter), and additionally lets
// the test thread block until a specific number of calls have *started*
// (before `hold` elapses) -- needed here to deterministically enqueue the
// Delete/second-Insert only once the first Insert is confirmably already
// executing, rather than racing against the planner picking it up.
class IntervalRecordingAdapter : public IAdapter {
 public:
	explicit IntervalRecordingAdapter(std::chrono::milliseconds hold) : hold_(hold) {}

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

	std::vector<CallInterval> intervals() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return intervals_;
	}

	// Blocks until at least `count` calls have been *entered* (i.e. are
	// somewhere inside their `hold_` sleep, definitely already dispatched
	// and executing on a worker) -- see class doc comment.
	void waitForCallsStarted(std::size_t count) {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [&] { return calls_started_ >= count; });
	}

 private:
	void record(ScheduledKind kind, std::optional<ModifyOp> op) {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			++calls_started_;
			cv_.notify_all();
		}
		Clock::time_point start = Clock::now();
		std::this_thread::sleep_for(hold_);
		Clock::time_point end = Clock::now();
		std::lock_guard<std::mutex> lock(mutex_);
		intervals_.push_back({kind, op, start, end});
	}

	std::chrono::milliseconds hold_;
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::size_t calls_started_ = 0;
	std::vector<CallInterval> intervals_;
};

TEST(OpSchedulerFairnessTest, DoesNotBlockOnAnInadmissibleChoiceWhenAnAdmissibleAlternativeExists) {
	IntervalRecordingAdapter adapter(std::chrono::milliseconds(60));

	SchedulingConfig config;
	config.traverse_batch_size = 1;
	config.modify_batch_size = 1;
	config.max_execution_threads = 2;  // both Inserts need a worker each to run concurrently

	OpScheduler scheduler(config);
	scheduler.start(adapter);

	// First Insert -- once this is confirmably executing (inflight_modify_
	// == 1, inflight_modify_op_ == Insert on the real OpScheduler), queue_ is
	// otherwise empty, matching the scenario this test targets.
	ModifyRequest insert1;
	insert1.op = ModifyOp::Insert;
	insert1.target = 1;
	std::future<ModifyResult> insert1_future = scheduler.schedule(insert1);
	adapter.waitForCallsStarted(1);

	// Enqueue Delete *before* the second Insert, so queue_.front() is the
	// Delete -- exactly the ordering the old queue_.front()-based op
	// selection would have committed to and then blocked on, even though
	// the Insert right behind it was safe to run concurrently with the
	// first Insert the whole time.
	ModifyRequest delete_request;
	delete_request.op = ModifyOp::Delete;
	delete_request.target = 2;
	std::future<ModifyResult> delete_future = scheduler.schedule(delete_request);

	ModifyRequest insert2;
	insert2.op = ModifyOp::Insert;
	insert2.target = 3;
	std::future<ModifyResult> insert2_future = scheduler.schedule(insert2);

	insert1_future.get();
	delete_future.get();
	insert2_future.get();
	scheduler.shutdown();

	std::vector<CallInterval> intervals = adapter.intervals();
	ASSERT_EQ(intervals.size(), 3u);

	const CallInterval* insert1_interval = nullptr;
	const CallInterval* insert2_interval = nullptr;
	const CallInterval* delete_interval = nullptr;
	for (const CallInterval& interval : intervals) {
		if (interval.op == ModifyOp::Delete) {
			delete_interval = &interval;
		} else if (insert1_interval == nullptr) {
			insert1_interval = &interval;
		} else {
			insert2_interval = &interval;
		}
	}
	ASSERT_NE(insert1_interval, nullptr);
	ASSERT_NE(insert2_interval, nullptr);
	ASSERT_NE(delete_interval, nullptr);

	EXPECT_TRUE(Overlaps(*insert1_interval, *insert2_interval))
			<< "the second Insert never overlapped with the first -- it was blocked behind the queued Delete instead of "
				 "running concurrently with the same-op Insert that was already admissible";
	EXPECT_FALSE(Overlaps(*insert1_interval, *delete_interval))
			<< "Delete overlapped with an in-flight Insert -- the isolation rule itself was violated, not just a "
				 "throughput regression";
	EXPECT_FALSE(Overlaps(*insert2_interval, *delete_interval))
			<< "Delete overlapped with an in-flight Insert -- the isolation rule itself was violated, not just a "
				 "throughput regression";
}

// Always prefers Traverse whenever any is pending, regardless of how long
// something else has waited -- delegates actual candidate selection to a
// plain FifoSchedulingPolicy, since only chooseBatchKind()'s preference is
// what this test needs to control.
class AlwaysPreferTraversePolicy : public SchedulingPolicy {
 public:
	ScheduledKind chooseBatchKind(const ScheduledOperationQueue&) const override { return ScheduledKind::Traverse; }

	std::optional<std::size_t> selectCandidateIndex(const ScheduledOperationQueue& queue, ScheduledKind batch_kind,
																									const ScheduledOperationBatch& current_batch) const override {
		return fifo_.selectCandidateIndex(queue, batch_kind, current_batch);
	}

	bool canAppendToBatch(ScheduledKind batch_kind, const ScheduledOperation& candidate,
												const ScheduledOperationBatch& current_batch) const override {
		return fifo_.canAppendToBatch(batch_kind, candidate, current_batch);
	}

 private:
	FifoSchedulingPolicy fifo_;
};

// Records the arrival order of Traverse/Insert calls, plus a synchronization
// "gate": a Delete request targeting kGateTarget blocks inside modifyHost()
// until the test explicitly releases it. Both tests below submit Insert and
// one Traverse *while the gate holds*, so neither is dispatched prematurely
// -- both sit genuinely inadmissible in queue_ together (blocked by the
// in-flight gate Delete, not by AlwaysPreferTraversePolicy's own
// preference) -- and become admissible at exactly the same instant once the
// gate releases. This is the only way to get a real, non-racy "both
// admissible, which wins" decision: without the gate, any freshly-submitted
// Traverse would simply get admitted immediately on its own (Traverse-vs-
// Traverse is unconditionally concurrent, regardless of how much else is
// already in flight), so it would never sit alongside a waiting Insert long
// enough for the two to be genuinely compared.
class OrderRecordingAdapter : public IAdapter {
 public:
	static constexpr VectorId kGateTarget = 999;

	std::vector<TraverseResult> traverseHost(const std::vector<TraverseRequest>& requests) override {
		record("Traverse");
		return std::vector<TraverseResult>(requests.size());
	}
	std::vector<ModifyResult> modifyHost(const std::vector<ModifyRequest>& requests) override {
		const ModifyRequest& request = requests.front();
		if (request.op == ModifyOp::Delete && request.target == kGateTarget) {
			std::unique_lock<std::mutex> lock(gate_mutex_);
			gate_entered_ = true;
			gate_cv_.notify_all();
			gate_cv_.wait(lock, [this] { return gate_release_; });
		} else {
			record("Insert");
		}
		std::vector<ModifyResult> results(requests.size());
		for (ModifyResult& result : results) result.ok = true;
		return results;
	}
	IRegion* resolveRegion(RegionId) override { return nullptr; }
	std::vector<RegionId> allRegions() const override { return {}; }

	void waitForGateEntered() {
		std::unique_lock<std::mutex> lock(gate_mutex_);
		gate_cv_.wait(lock, [this] { return gate_entered_; });
	}
	void releaseGate() {
		std::lock_guard<std::mutex> lock(gate_mutex_);
		gate_release_ = true;
		gate_cv_.notify_all();
	}
	std::vector<std::string> order() const {
		std::lock_guard<std::mutex> lock(order_mutex_);
		return order_;
	}

 private:
	void record(const char* label) {
		std::lock_guard<std::mutex> lock(order_mutex_);
		order_.push_back(label);
	}

	std::mutex gate_mutex_;
	std::condition_variable gate_cv_;
	bool gate_entered_ = false;
	bool gate_release_ = false;

	mutable std::mutex order_mutex_;
	std::vector<std::string> order_;
};

TEST(OpSchedulerFairnessTest, StarvationOverrideDisabledRespectsPolicyPreferenceEvenWhenSomethingElseIsOverdue) {
	OrderRecordingAdapter adapter;
	SchedulingConfig config;  // starvation_threshold defaults to 0 (disabled)
	config.max_execution_threads = 2;

	OpScheduler scheduler(config, std::make_unique<AlwaysPreferTraversePolicy>());
	scheduler.start(adapter);

	ModifyRequest gate;
	gate.op = ModifyOp::Delete;
	gate.target = OrderRecordingAdapter::kGateTarget;
	std::future<ModifyResult> gate_future = scheduler.schedule(gate);
	adapter.waitForGateEntered();

	// Both genuinely inadmissible right now (the gate Delete is in flight) --
	// they accumulate in queue_ together rather than racing to be dispatched.
	ModifyRequest insert;
	insert.op = ModifyOp::Insert;
	insert.target = 1;
	std::future<ModifyResult> insert_future = scheduler.schedule(insert);
	std::future<TraverseResult> traverse_future = scheduler.schedule(TraverseRequest{});

	adapter.releaseGate();
	gate_future.get();
	insert_future.get();
	traverse_future.get();
	scheduler.shutdown();

	std::vector<std::string> order = adapter.order();
	ASSERT_EQ(order.size(), 2u);
	EXPECT_EQ(order.front(), "Traverse")
			<< "AlwaysPreferTraversePolicy's preference should have been honored (starvation_threshold disabled) -- "
				 "Insert ran first instead, which only the override (not policy) should be capable of";
}

TEST(OpSchedulerFairnessTest, StarvationOverrideEnabledForcesTheOverdueClassRegardlessOfPolicyPreference) {
	OrderRecordingAdapter adapter;
	SchedulingConfig config;
	config.max_execution_threads = 2;
	config.starvation_threshold = std::chrono::milliseconds(10);

	OpScheduler scheduler(config, std::make_unique<AlwaysPreferTraversePolicy>());
	scheduler.start(adapter);

	ModifyRequest gate;
	gate.op = ModifyOp::Delete;
	gate.target = OrderRecordingAdapter::kGateTarget;
	std::future<ModifyResult> gate_future = scheduler.schedule(gate);
	adapter.waitForGateEntered();

	ModifyRequest insert;
	insert.op = ModifyOp::Insert;
	insert.target = 1;
	std::future<ModifyResult> insert_future = scheduler.schedule(insert);
	std::future<TraverseResult> traverse_future = scheduler.schedule(TraverseRequest{});

	// Insert's enqueued_at was captured above, before the gate even opens --
	// let it age past starvation_threshold while the gate still holds, so the
	// very first decision after release already finds it overdue.
	std::this_thread::sleep_for(std::chrono::milliseconds(30));

	adapter.releaseGate();
	gate_future.get();
	insert_future.get();
	traverse_future.get();
	scheduler.shutdown();

	std::vector<std::string> order = adapter.order();
	ASSERT_EQ(order.size(), 2u);
	EXPECT_EQ(order.front(), "Insert")
			<< "starvation_threshold was configured (10ms) and Insert had already waited well past it -- the override "
				 "should have forced it to be scheduled first despite AlwaysPreferTraversePolicy's bias toward Traverse";
}

}  // namespace
