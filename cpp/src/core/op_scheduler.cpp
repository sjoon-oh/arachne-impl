#include "core/op_scheduler.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "logging.hpp"
#include "telemetry/trace.hpp"

namespace arachne {

OpScheduler::OpScheduler(SchedulingConfig config, std::unique_ptr<SchedulingPolicy> policy)
	: traverse_batch_size_(sanitizeBatchSize(config.traverse_batch_size)),
	  modify_batch_size_(sanitizeBatchSize(config.modify_batch_size)),
	  max_execution_threads_(sanitizeThreadCount(config.max_execution_threads)),
	  batch_wait_timeout_(config.batch_wait_timeout),
	  starvation_threshold_(config.starvation_threshold),
	  policy_(std::move(policy)) {
	if (policy_ == nullptr) {
		policy_ = std::make_unique<FifoSchedulingPolicy>();
	}
}

OpScheduler::~OpScheduler() { shutdown(); }

void OpScheduler::start(IAdapter& adapter, std::function<void(std::size_t)> on_worker_start,
		std::function<void(TraverseRequest&)> prepare_traverse,
		std::function<void(ModifyRequest&)> prepare_modify) {
	std::scoped_lock lock(mutex_);
	if (running_) {
		throw std::logic_error("OpScheduler already started");
	}
	adapter_ = &adapter;
	on_worker_start_ = std::move(on_worker_start);
	prepare_traverse_ = std::move(prepare_traverse);
	prepare_modify_ = std::move(prepare_modify);
	traverse_modify_isolation_enabled_ = adapter_->requiresTraverseModifyIsolation();
	stop_requested_ = false;
	running_ = true;
	planner_ = std::thread(&OpScheduler::plannerLoop, this);
	execution_workers_.reserve(max_execution_threads_);
	for (std::size_t i = 0; i < max_execution_threads_; ++i) {
		execution_workers_.emplace_back(&OpScheduler::workerLoop, this, i);
	}
	ARACHNE_LOG_INFO(
			"OpScheduler::start: {} execution worker(s), traverse_batch_size={} modify_batch_size={} "
			"traverse_modify_isolation_enabled={}",
			max_execution_threads_, traverse_batch_size_, modify_batch_size_, traverse_modify_isolation_enabled_);
}

void OpScheduler::shutdown() {
	{
		std::scoped_lock lock(mutex_);
		if (!running_) {
			return;
		}
		stop_requested_ = true;
		cv_incoming_.notify_all();
		cv_dispatch_.notify_all();
	}
	ARACHNE_LOG_INFO("OpScheduler::shutdown: stop requested, joining worker threads");

	if (planner_.joinable()) {
		planner_.join();
	}
	for (auto& worker : execution_workers_) {
		if (worker.joinable()) {
			worker.join();
		}
	}

	std::scoped_lock lock(mutex_);
	running_ = false;
	stop_requested_ = false;
	adapter_ = nullptr;
	on_worker_start_ = nullptr;
	execution_workers_.clear();
	// Should already be back to this state -- every dispatched batch's
	// releaseExecutionSlot() runs before its worker loops back, and both
	// planner_/execution_workers_ are fully joined above -- but reset
	// explicitly so a subsequent start() on this same instance never inherits
	// stale gate state.
	inflight_traverse_ = 0;
	inflight_modify_ = 0;
	inflight_modify_op_.reset();
	ARACHNE_LOG_INFO("OpScheduler::shutdown: all worker threads joined");
}

std::future<TraverseResult> OpScheduler::schedule(
		TraverseRequest request, std::function<void(std::exception_ptr, const TraverseResult&)> on_complete) {
	std::promise<TraverseResult> promise;
	auto future = promise.get_future();

	{
		std::scoped_lock lock(mutex_);
		if (!running_ || stop_requested_ || adapter_ == nullptr) {
			promise.set_exception(std::make_exception_ptr(
					std::runtime_error("OpScheduler is not running")));
			return future;
		}

		queue_.emplace_back(TraverseTask{next_id_++, std::chrono::steady_clock::now(), std::move(request),
																	 std::move(promise), std::move(on_complete)});
	}

	cv_incoming_.notify_one();
	return future;
}

std::future<ModifyResult> OpScheduler::schedule(
		ModifyRequest request, std::function<void(std::exception_ptr, const ModifyResult&)> on_complete) {
	std::promise<ModifyResult> promise;
	auto future = promise.get_future();

	{
		std::scoped_lock lock(mutex_);
		if (!running_ || stop_requested_ || adapter_ == nullptr) {
			promise.set_exception(std::make_exception_ptr(
					std::runtime_error("OpScheduler is not running")));
			return future;
		}

		queue_.emplace_back(ModifyTask{next_id_++, std::chrono::steady_clock::now(), std::move(request),
																	 std::move(promise), std::move(on_complete)});
	}

	cv_incoming_.notify_one();
	return future;
}

void OpScheduler::setTraverseBatchSize(std::size_t size) {
	setBatchSizeValue(ScheduledKind::Traverse, size);
}

void OpScheduler::setModifyBatchSize(std::size_t size) {
	setBatchSizeValue(ScheduledKind::Modify, size);
}

void OpScheduler::setMaxExecutionThreads(std::size_t threads) {
	setExecutionThreadValue(threads);
}

std::size_t OpScheduler::traverseBatchSize() const {
	return batchSizeValue(ScheduledKind::Traverse);
}

std::size_t OpScheduler::modifyBatchSize() const {
	return batchSizeValue(ScheduledKind::Modify);
}

std::size_t OpScheduler::maxExecutionThreads() const {
	std::scoped_lock lock(mutex_);
	return max_execution_threads_;
}

void OpScheduler::plannerLoop() {
	while (true) {
		std::unique_lock lock(mutex_);
		cv_incoming_.wait(lock, [this] { return stop_requested_ || !queue_.empty(); });

		if (stop_requested_ && queue_.empty()) {
			break;
		}

		if (queue_.empty()) {
			continue;
		}

		std::optional<PendingClass> chosen = selectNextBatchKind(admissibleClasses());

		if (!chosen.has_value()) {
			// Every (kind, op) class currently represented in queue_ conflicts
			// with whatever's already in flight (see class doc comment) -- a
			// genuine floor, not a scheduling gap: no reordering could make any
			// of this safe to run right now. Wait for in-flight execution state
			// to change -- releaseExecutionSlot() notifies cv_incoming_ too --
			// then loop back to the top and re-decide from scratch.
			cv_incoming_.wait(lock, [this] { return stop_requested_ || !admissibleClasses().empty(); });
			continue;
		}

		// Seed the batch with exactly the class selectNextBatchKind() chose --
		// collectBatch()'s own SchedulingPolicy-driven loop then fills the rest,
		// naturally constrained to the same mode/op via canAppendToBatch()'s
		// existing homogeneity check against this seed (current_batch.front()).
		ScheduledOperationBatch batch;
		batch.push_back(extractOldest(chosen->kind, chosen->op));

		std::size_t batch_target = targetBatchSizeFor(chosen->kind);
		collectBatch(batch, chosen->kind, batch_target, lock);

		// Never empty -- extractOldest() above always seeds at least one op.
		reserveExecutionSlot(chosen->kind, chosen->op);
		dispatch_queue_.push_back(std::move(batch));
		cv_dispatch_.notify_one();
	}
}

std::vector<OpScheduler::PendingClass> OpScheduler::admissibleClasses() const {
	std::vector<PendingClass> result;

	auto consider = [&](ScheduledKind kind, std::optional<ModifyOp> op) {
		for (const ScheduledOperation& entry : queue_) {
			if (kindOf(entry) != kind) continue;
			if (kind == ScheduledKind::Modify && std::get<ModifyTask>(entry).request.op != *op) continue;
			// First match in arrival order (queue_ is a deque, preserving
			// enqueue order) is this class's oldest pending member.
			if (canAdmit(kind, op)) {
				std::chrono::steady_clock::time_point enqueued_at = kind == ScheduledKind::Traverse
																															 ? std::get<TraverseTask>(entry).enqueued_at
																															 : std::get<ModifyTask>(entry).enqueued_at;
				result.push_back(PendingClass{kind, op, enqueued_at});
			}
			return;
		}
	};

	consider(ScheduledKind::Traverse, std::nullopt);
	consider(ScheduledKind::Modify, ModifyOp::Insert);
	consider(ScheduledKind::Modify, ModifyOp::Delete);
	return result;
}

std::optional<OpScheduler::PendingClass> OpScheduler::selectNextBatchKind(
		const std::vector<PendingClass>& admissible) const {
	if (admissible.empty()) return std::nullopt;

	auto oldestOf = [](const std::vector<PendingClass>& candidates) -> const PendingClass& {
		return *std::min_element(candidates.begin(), candidates.end(), [](const PendingClass& a, const PendingClass& b) {
			return a.oldest_enqueued_at < b.oldest_enqueued_at;
		});
	};

	// 1. Starvation override -- see class doc comment. Disabled (skipped
	// entirely) when starvation_threshold_ is zero, the default.
	if (starvation_threshold_.count() > 0) {
		const PendingClass& oldest = oldestOf(admissible);
		if (std::chrono::steady_clock::now() - oldest.oldest_enqueued_at >= starvation_threshold_) {
			return oldest;
		}
	}

	// 2. SchedulingPolicy's own preference, honored whenever it's actually
	// admissible right now -- full authority over the common case. For
	// Modify, prefer whichever op has waited longer if both happen to be
	// admissible simultaneously (only possible when nothing is in flight).
	ScheduledKind preferred = policy_->chooseBatchKind(queue_);
	std::vector<PendingClass> preferred_matches;
	for (const PendingClass& candidate : admissible) {
		if (candidate.kind == preferred) preferred_matches.push_back(candidate);
	}
	if (!preferred_matches.empty()) return oldestOf(preferred_matches);

	// 3. SchedulingPolicy's preferred kind isn't currently admissible at all,
	// but something else is -- rather than block with idle workers while
	// perfectly runnable work waits (see class doc comment), fall back to
	// whichever admissible class has waited longest.
	return oldestOf(admissible);
}

ScheduledOperation OpScheduler::extractOldest(ScheduledKind kind, std::optional<ModifyOp> op) {
	for (auto it = queue_.begin(); it != queue_.end(); ++it) {
		if (kindOf(*it) != kind) continue;
		if (kind == ScheduledKind::Modify && std::get<ModifyTask>(*it).request.op != *op) continue;
		ScheduledOperation result = std::move(*it);
		queue_.erase(it);
		return result;
	}
	// admissibleClasses()/selectNextBatchKind() already confirmed a matching
	// entry exists, under the same uninterrupted mutex_ hold plannerLoop()
	// still holds here -- reaching this means that invariant broke.
	throw std::logic_error("OpScheduler::extractOldest: no matching queue_ entry (invariant violation)");
}

void OpScheduler::workerLoop(std::size_t worker_index) {
	if (on_worker_start_) on_worker_start_(worker_index);
	while (true) {
		ScheduledOperationBatch batch;
		{
			std::unique_lock lock(mutex_);
			cv_dispatch_.wait(lock, [this] { return stop_requested_ || !dispatch_queue_.empty(); });

			if (stop_requested_ && dispatch_queue_.empty()) {
				break;
			}

			if (dispatch_queue_.empty()) {
				continue;
			}

			batch = std::move(dispatch_queue_.front());
			dispatch_queue_.pop_front();
		}
		// Captured before executeBatch() moves-from batch -- reserveExecutionSlot()
		// (plannerLoop()) already recorded this same (kind, op) when the batch was
		// built, so releasing by kind alone is enough to find the right counter.
		ScheduledKind kind = kindOf(batch.front());
		executeBatch(std::move(batch));
		releaseExecutionSlot(kind);
	}
}

ScheduledKind OpScheduler::kindOf(const ScheduledOperation& op) const {
	if (std::holds_alternative<TraverseTask>(op)) {
		return ScheduledKind::Traverse;
	}
	return ScheduledKind::Modify;
}

std::size_t OpScheduler::targetBatchSizeFor(ScheduledKind kind) const {
	// Reads the fields directly rather than through the locking
	// batchSizeValue() -- only called from plannerLoop(), which already holds
	// mutex_ for the whole loop body; re-locking the non-recursive mutex_
	// would deadlock.
	return kind == ScheduledKind::Traverse ? traverse_batch_size_ : modify_batch_size_;
}

void OpScheduler::collectBatch(ScheduledOperationBatch& batch, ScheduledKind batch_kind,
															std::size_t batch_target,
															std::unique_lock<OpSchedulerMutex>& lock) {
	while (batch.size() < batch_target) {
		if (queue_.empty()) {
			if (batch.empty() || batch_wait_timeout_.count() == 0) {
				break;
			}

			cv_incoming_.wait_for(lock, batch_wait_timeout_,
													 [this] { return stop_requested_ || !queue_.empty(); });
			if (queue_.empty() || stop_requested_) {
				break;
			}
		}

			auto candidate_index = policy_->selectCandidateIndex(queue_, batch_kind, batch);
			if (!candidate_index.has_value()) {
				break;
			}

			auto iter = queue_.begin() + static_cast<std::ptrdiff_t>(*candidate_index);
			batch.push_back(std::move(*iter));
			queue_.erase(iter);
		}
}

bool OpScheduler::canAdmit(ScheduledKind kind, std::optional<ModifyOp> op) const {
	if (!traverse_modify_isolation_enabled_) return true;

	if (kind == ScheduledKind::Traverse) {
		return inflight_modify_ == 0;
	}

	// Modify: blocked by any in-flight Traverse, or by an in-flight Modify of
	// a *different* op -- same-op Modify batches may still run concurrently
	// (see this class's doc comment).
	if (inflight_traverse_ > 0) return false;
	if (inflight_modify_ > 0 && inflight_modify_op_ != op) return false;
	return true;
}

void OpScheduler::reserveExecutionSlot(ScheduledKind kind, std::optional<ModifyOp> op) {
	if (!traverse_modify_isolation_enabled_) return;

	if (kind == ScheduledKind::Traverse) {
		++inflight_traverse_;
		return;
	}
	inflight_modify_op_ = op;
	++inflight_modify_;
}

void OpScheduler::releaseExecutionSlot(ScheduledKind kind) {
	if (!traverse_modify_isolation_enabled_) return;

	{
		std::scoped_lock lock(mutex_);
		if (kind == ScheduledKind::Traverse) {
			--inflight_traverse_;
		} else {
			--inflight_modify_;
			if (inflight_modify_ == 0) inflight_modify_op_.reset();
		}
	}
	// Wakes the planner if it's blocked in canAdmit()'s wait (plannerLoop()),
	// same condition variable schedule() already uses for new arrivals.
	cv_incoming_.notify_all();
}

void OpScheduler::executeBatch(ScheduledOperationBatch batch) {
	if (batch.empty()) return;

	// SchedulingPolicy::canAppendToBatch() guarantees every collectBatch()
	// result is kind-homogeneous (see its doc comment), so the front
	// element's kind tells us which of the two batch paths applies to the
	// whole thing.
	if (kindOf(batch.front()) == ScheduledKind::Traverse) {
		executeTraverseBatch(std::move(batch));
	} else {
		executeModifyBatch(std::move(batch));
	}
}

void OpScheduler::executeTraverseBatch(ScheduledOperationBatch batch) {
	ARACHNE_TRACE_SCOPE("OpScheduler", "executeTraverseBatch");
	std::vector<TraverseRequest> requests;
	requests.reserve(batch.size());
	for (auto& op : batch) requests.push_back(std::get<TraverseTask>(op).request);
	if (prepare_traverse_) {
		for (TraverseRequest& request : requests) prepare_traverse_(request);
	}

	// Routing hints can become stale independently. Keep adapter calls batch
	// homogeneous by demoting the entire batch if any request failed its
	// execution-time residency validation.
	bool device = true;
	for (const TraverseRequest& request : requests) {
		if (request.mode != ExecutionMode::GpuOnly) device = false;
	}
	if (!device) {
		for (TraverseRequest& request : requests) {
			request.mode = ExecutionMode::Hybrid;
			request.residency_pin.reset();
		}
	}

	try {
		std::vector<TraverseResult> results =
				device ? adapter_->traverseDevice(requests) : adapter_->traverseHost(requests);
		if (results.size() != batch.size()) {
			throw std::logic_error(
					"IAdapter::traverseHost/traverseDevice: result count does not match request count");
		}
		for (std::size_t i = 0; i < batch.size(); ++i) {
			results[i].execution_mode = requests[i].mode;
			TraverseTask& task = std::get<TraverseTask>(batch[i]);
			// Runs on this worker thread, before set_value() -- see
			// schedule()'s doc comment for the ordering guarantee this gives a
			// caller blocked on the returned future.
			if (task.on_complete) task.on_complete(nullptr, results[i]);
			task.promise.set_value(std::move(results[i]));
		}
	} catch (...) {
		ARACHNE_LOG_ERROR(
				"OpScheduler::executeTraverseBatch: adapter threw for a batch of {} request(s) (device={}) -- "
				"propagating exception to every future in the batch",
				batch.size(), device);
		std::exception_ptr eptr = std::current_exception();
		for (auto& op : batch) {
			TraverseTask& task = std::get<TraverseTask>(op);
			// See TraverseTask::on_complete's doc comment: fires on failure too,
			// before the promise's own set_exception() below, so a chained
			// continuation is never left waiting on a promise nobody resolves.
			if (task.on_complete) {
				try {
					task.on_complete(eptr, TraverseResult{});
				} catch (...) {
				}
			}
			try {
				task.promise.set_exception(eptr);
			} catch (...) {
			}
		}
	}
}

void OpScheduler::executeModifyBatch(ScheduledOperationBatch batch) {
	ARACHNE_TRACE_SCOPE("OpScheduler", "executeModifyBatch");
	std::vector<ModifyRequest> requests;
	requests.reserve(batch.size());
	for (auto& op : batch) requests.push_back(std::get<ModifyTask>(op).request);
	if (prepare_modify_) {
		for (ModifyRequest& request : requests) prepare_modify_(request);
	}

	bool device = true;
	for (const ModifyRequest& request : requests) {
		if (request.mode != ExecutionMode::GpuOnly) device = false;
	}
	if (!device) {
		for (ModifyRequest& request : requests) {
			request.mode = ExecutionMode::Hybrid;
			request.lease = LeaseHandle{};
			request.residency_pin.reset();
		}
	}

	try {
		std::vector<ModifyResult> results =
				device ? adapter_->modifyDevice(requests) : adapter_->modifyHost(requests);
		if (results.size() != batch.size()) {
			throw std::logic_error(
					"IAdapter::modifyHost/modifyDevice: result count does not match request count");
		}
		for (std::size_t i = 0; i < batch.size(); ++i) {
			results[i].execution_mode = requests[i].mode;
			ModifyTask& task = std::get<ModifyTask>(batch[i]);
			if (task.on_complete) task.on_complete(nullptr, results[i]);
			task.promise.set_value(std::move(results[i]));
		}
	} catch (...) {
		ARACHNE_LOG_ERROR(
				"OpScheduler::executeModifyBatch: adapter threw for a batch of {} request(s) (device={}) -- "
				"propagating exception to every future in the batch",
				batch.size(), device);
		std::exception_ptr eptr = std::current_exception();
		for (auto& op : batch) {
			ModifyTask& task = std::get<ModifyTask>(op);
			if (task.on_complete) {
				try {
					task.on_complete(eptr, ModifyResult{});
				} catch (...) {
				}
			}
			try {
				task.promise.set_exception(eptr);
			} catch (...) {
			}
		}
	}
}

void OpScheduler::setBatchSizeValue(ScheduledKind kind, std::size_t size) {
	std::scoped_lock lock(mutex_);
	size = sanitizeBatchSize(size);
	switch (kind) {
		case ScheduledKind::Traverse:
			traverse_batch_size_ = size;
			break;
		case ScheduledKind::Modify:
			modify_batch_size_ = size;
			break;
	}
}

void OpScheduler::setExecutionThreadValue(std::size_t threads) {
	std::scoped_lock lock(mutex_);
	if (running_) {
		throw std::logic_error("cannot change execution thread count while scheduler is running");
	}
	max_execution_threads_ = sanitizeThreadCount(threads);
}

std::size_t OpScheduler::batchSizeValue(ScheduledKind kind) const {
	std::scoped_lock lock(mutex_);
	return kind == ScheduledKind::Traverse ? traverse_batch_size_ : modify_batch_size_;
}

std::size_t OpScheduler::sanitizeBatchSize(std::size_t size) const {
	return size == 0 ? 1 : size;
}

std::size_t OpScheduler::sanitizeThreadCount(std::size_t threads) const {
	return threads == 0 ? 1 : threads;
}

}  // namespace arachne
