#include "core/op_scheduler.hpp"

#include <stdexcept>
#include <utility>

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

void OpScheduler::start(IndexAdapter& adapter) {
	std::scoped_lock lock(mutex_);
	if (running_) {
		throw std::logic_error("OpScheduler already started");
	}
	adapter_ = &adapter;
	stop_requested_ = false;
	running_ = true;
	planner_ = std::thread(&OpScheduler::plannerLoop, this);
	execution_workers_.reserve(max_execution_threads_);
	for (std::size_t i = 0; i < max_execution_threads_; ++i) {
		execution_workers_.emplace_back(&OpScheduler::workerLoop, this);
	}
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
	execution_workers_.clear();
}

std::future<TraverseResult> OpScheduler::schedule(TraverseRequest request) {
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
																	 std::move(promise)});
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
		ScheduledOperationBatch batch;
		std::size_t batch_target = targetBatchSizeFor(batch_kind);
		collectBatch(batch, batch_kind, batch_target, lock);

		if (!batch.empty()) {
			dispatch_queue_.push_back(std::move(batch));
			cv_dispatch_.notify_one();
		}
	}
}

void OpScheduler::workerLoop() {
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
		executeBatch(std::move(batch));
	}
}

ScheduledKind OpScheduler::kindOf(const ScheduledOperation& op) const {
	if (std::holds_alternative<TraverseTask>(op)) {
		return ScheduledKind::Traverse;
	}
	return ScheduledKind::Modify;
}

std::size_t OpScheduler::targetBatchSizeFor(ScheduledKind kind) const {
	return kind == ScheduledKind::Traverse ? batchSizeValue(ScheduledKind::Traverse)
																				: batchSizeValue(ScheduledKind::Modify);
}

void OpScheduler::collectBatch(ScheduledOperationBatch& batch, ScheduledKind batch_kind,
															std::size_t batch_target,
															std::unique_lock<std::mutex>& lock) {
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

void OpScheduler::executeBatch(ScheduledOperationBatch batch) {
	if (batch.empty()) return;

	for (auto& op : batch) {
		if (kindOf(op) == ScheduledKind::Traverse) {
			auto& task = std::get<TraverseTask>(op);
			executeTask(task);
		} else {
			auto& task = std::get<ModifyTask>(op);
			executeTask(task);
		}
	}
}

void OpScheduler::executeTask(TraverseTask& task) {
	try {
		task.promise.set_value(adapter_->traverse(task.request));
	} catch (...) {
		try {
			task.promise.set_exception(std::current_exception());
		} catch (...) {
		}
	}
}

void OpScheduler::executeTask(ModifyTask& task) {
	try {
		task.promise.set_value(adapter_->modify(task.request));
	} catch (...) {
		try {
			task.promise.set_exception(std::current_exception());
		} catch (...) {
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
