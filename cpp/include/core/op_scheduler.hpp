#pragma once

#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "adapter/index_adapter.hpp"
#include "core/scheduling_policy.hpp"

namespace arachne {

/// External tuning knobs for Arachne's operation scheduling/batching layer.
/// This layer intentionally stays index-agnostic: it only sees
/// TraverseRequest/ModifyRequest and orchestrates when/how to call
/// IAdapter's Host/Device entry points.
struct SchedulingConfig {
	/// Maximum number of same-kind ops grouped together before dispatch.
	/// Traverse and modify each have independent knobs.
	std::size_t traverse_batch_size = 1;
	std::size_t modify_batch_size = 1;

	/// Maximum number of execution worker threads. This controls how many
	/// execution batches can run in parallel.
	std::size_t max_execution_threads = 1;

	/// Optional micro-batching window. If > 0, after at least one op is in the
	/// hand-built dispatch batch, the scheduler waits up to this long for
	/// additional eligible operations to join the same batch.
	std::chrono::microseconds batch_wait_timeout{0};
};

	/// Arachne index-agnostic traversal/modify scheduler.
/// Stage-1 accepts index requests, stage-2 reorders/batches, and stage-3 executes
/// batches in an execution thread pool.
class OpScheduler {
 public:
  explicit OpScheduler(SchedulingConfig config = {},
                       std::unique_ptr<SchedulingPolicy> policy = nullptr);
  ~OpScheduler();

  OpScheduler(const OpScheduler&) = delete;
  OpScheduler& operator=(const OpScheduler&) = delete;

  /// Connects this scheduler to an adapter. Must be called before schedule().
  /// `on_worker_start`, if provided, is invoked exactly once on each
  /// execution worker thread -- before it processes any batch -- with that
  /// worker's 0-based index (< maxExecutionThreads()). OpScheduler never
  /// interprets the index or the callback itself; it stays index-/
  /// GPU-agnostic on purpose. Controller uses this hook to bind each worker
  /// thread to its own dedicated CUDA stream (see
  /// gpu::DeviceContext::workerStream()) via thread-local state, so a
  /// GPU-native traverseDevice()/modifyDevice() call running on that thread
  /// can pick up the right stream without OpScheduler needing to know CUDA
  /// exists at all.
  void start(IAdapter& adapter, std::function<void(std::size_t)> on_worker_start = nullptr);

  /// Stops worker processing and drains queues before shutdown.
  void shutdown();

  /// Schedules one Traverse request. Returns a future that becomes ready once
  /// execution finishes. `on_complete`, if provided, is invoked on the
  /// execution worker thread that actually computed the result -- right
  /// before the returned future is made ready, so it is guaranteed to have
  /// already run by the time a caller's future.get() unblocks -- not on
  /// whichever thread calls future.get(). This is what lets a caller move
  /// result-dependent bookkeeping (that would otherwise contend on a shared
  /// lock from every calling thread) onto the bounded set of worker threads
  /// instead. OpScheduler never interprets `on_complete` itself, same as
  /// start()'s on_worker_start.
  std::future<TraverseResult> schedule(TraverseRequest request,
                                        std::function<void(const TraverseResult&)> on_complete = nullptr);

  /// Schedules one Modify request. Returns a future that becomes ready once
  /// execution finishes.
  std::future<ModifyResult> schedule(ModifyRequest request);

  /// Runtime-adjustable knobs for callers that want to tune behavior.
  void setTraverseBatchSize(std::size_t size);
  void setModifyBatchSize(std::size_t size);
  /// Must be called before start(); changing this while running throws.
  void setMaxExecutionThreads(std::size_t threads);

  std::size_t traverseBatchSize() const;
  std::size_t modifyBatchSize() const;
  std::size_t maxExecutionThreads() const;

private:
	void plannerLoop();
	void workerLoop(std::size_t worker_index);
	ScheduledKind kindOf(const ScheduledOperation& op) const;
	std::size_t targetBatchSizeFor(ScheduledKind kind) const;
	void collectBatch(ScheduledOperationBatch& batch, ScheduledKind batch_kind,
									 std::size_t batch_target, std::unique_lock<std::mutex>& lock);
	// Dispatches a whole (kind- and mode-homogeneous, see SchedulingPolicy::
	// canAppendToBatch()) batch to exactly one of IAdapter's
	// traverseHost()/traverseDevice() or modifyHost()/modifyDevice() in one
	// call, rather than looping one request at a time -- see those methods'
	// doc comments (adapter/index_adapter.hpp) for why a genuinely
	// batch-aware index needs this shape to get real GPU throughput out of a
	// batch.
	void executeBatch(ScheduledOperationBatch batch);
	void executeTraverseBatch(ScheduledOperationBatch batch);
	void executeModifyBatch(ScheduledOperationBatch batch);

	void setBatchSizeValue(ScheduledKind kind, std::size_t size);
	void setExecutionThreadValue(std::size_t threads);

	std::size_t batchSizeValue(ScheduledKind kind) const;
	std::size_t sanitizeBatchSize(std::size_t size) const;
	std::size_t sanitizeThreadCount(std::size_t threads) const;

	// Configuration
	std::size_t traverse_batch_size_;
	std::size_t modify_batch_size_;
	std::size_t max_execution_threads_;
	std::chrono::microseconds batch_wait_timeout_;

	// Strategy
	std::unique_ptr<SchedulingPolicy> policy_;

	// Set once in start(), before any worker thread is spawned -- safe to
	// read from workerLoop() without holding mutex_ (thread creation
	// establishes a happens-before edge for everything start() did first).
	std::function<void(std::size_t)> on_worker_start_;

	// Lifecycle / threading
	mutable std::mutex mutex_;
	std::condition_variable cv_incoming_;
	std::condition_variable cv_dispatch_;
	bool running_ = false;
	bool stop_requested_ = false;
	std::thread planner_;
	std::vector<std::thread> execution_workers_;

	// Pending work queues and ids
	ScheduledOperationQueue queue_;
	std::deque<ScheduledOperationBatch> dispatch_queue_;
	std::uint64_t next_id_ = 1;

	// Adapter call target (owned outside this class)
	IAdapter* adapter_ = nullptr;
};

}  // namespace arachne
