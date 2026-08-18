#include "core/op_scheduler.hpp"

#include <stdexcept>
#include <utility>

#include "logging.hpp"
#include "telemetry/trace.hpp"

namespace arachne {

OpScheduler::OpScheduler(SchedulingConfig config, std::unique_ptr<SchedulingPolicy> policy)
	: traverse_batch_size_(sanitizeBatchSize(config.traverse_batch_size)),
	  modify_batch_size_(sanitizeBatchSize(config.modify_batch_size)),
	  max_execution_threads_(sanitizeThreadCount(config.max_execution_threads)),
	  batch_wait_timeout_(config.batch_wait_timeout),
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

std::future<TraverseResult> OpScheduler::schedule(TraverseRequest request,
																									 std::function<void(const TraverseResult&)> on_complete) {
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

std::future<ModifyResult> OpScheduler::schedule(ModifyRequest request) {
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
																	 std::move(promise)});
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

		ScheduledKind batch_kind = policy_->chooseBatchKind(queue_);
		std::optional<ModifyOp> batch_op;
		if (batch_kind == ScheduledKind::Modify) {
			batch_op = std::get<ModifyTask>(queue_.front()).request.op;
		}

		if (!canAdmit(batch_kind, batch_op)) {
			// Something already in flight conflicts with the next batch this
			// policy would build (see IAdapter::requiresTraverseModifyIsolation()
			// and this class's own doc comment). Wait for in-flight execution
			// state to change -- releaseExecutionSlot() notifies cv_incoming_ too
			// -- then loop back to the top and re-decide from scratch: queue_
			// itself can only be *appended* to while blocked here (nothing else
			// ever removes from it but this thread), but re-deciding is cheap and
			// avoids leaning on that invariant.
			cv_incoming_.wait(lock, [this, batch_kind, batch_op] {
				return stop_requested_ || canAdmit(batch_kind, batch_op);
			});
			continue;
		}

		ScheduledOperationBatch batch;
		std::size_t batch_target = targetBatchSizeFor(batch_kind);
		collectBatch(batch, batch_kind, batch_target, lock);

		if (!batch.empty()) {
			reserveExecutionSlot(batch_kind, batch_op);
			dispatch_queue_.push_back(std::move(batch));
			cv_dispatch_.notify_one();
		}
	}
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
			if (task.on_complete) task.on_complete(results[i]);
			task.promise.set_value(std::move(results[i]));
		}
	} catch (...) {
		ARACHNE_LOG_ERROR(
				"OpScheduler::executeTraverseBatch: adapter threw for a batch of {} request(s) (device={}) -- "
				"propagating exception to every future in the batch",
				batch.size(), device);
		std::exception_ptr eptr = std::current_exception();
		for (auto& op : batch) {
			try {
				std::get<TraverseTask>(op).promise.set_exception(eptr);
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
			std::get<ModifyTask>(batch[i]).promise.set_value(std::move(results[i]));
		}
	} catch (...) {
		ARACHNE_LOG_ERROR(
				"OpScheduler::executeModifyBatch: adapter threw for a batch of {} request(s) (device={}) -- "
				"propagating exception to every future in the batch",
				batch.size(), device);
		std::exception_ptr eptr = std::current_exception();
		for (auto& op : batch) {
			try {
				std::get<ModifyTask>(op).promise.set_exception(eptr);
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
