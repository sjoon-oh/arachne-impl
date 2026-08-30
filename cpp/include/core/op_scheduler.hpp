#pragma once

#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "adapter/index_adapter.hpp"
#include "core/scheduling_policy.hpp"
#include "telemetry/instrumented_mutex.hpp"

namespace arachne {

// See core/region_manager.hpp's own RegionManagerMutex/RegionManagerCondVar
// for the full rationale -- same pairing here, just scoped to OpScheduler's
// own mutex_/cv_incoming_/cv_dispatch_ instead ("OpScheduler-lockwait.csv"
// when ARACHNE_ENABLE_TRACING is on).
#ifdef ARACHNE_ENABLE_TRACING
class OpSchedulerMutex : public telemetry::InstrumentedMutex {
 public:
	OpSchedulerMutex() : InstrumentedMutex("OpScheduler") {}
};
using OpSchedulerCondVar = std::condition_variable_any;
#else
using OpSchedulerMutex = std::mutex;
using OpSchedulerCondVar = std::condition_variable;
#endif

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

	/// Bounds how long a currently-admissible (kind, op) class of pending
	/// work may sit unpicked while SchedulingPolicy keeps preferring
	/// something else, before the planner overrides that preference and
	/// forces it to be scheduled next -- see OpScheduler's class doc comment
	/// ("Starvation override"). 0 (the default) disables the override
	/// entirely, matching this feature not existing: a policy that never
	/// revisits a given (kind, op) can then withhold it indefinitely. This
	/// is a safety net for a *pluggable, non-FIFO* SchedulingPolicy --
	/// FifoSchedulingPolicy itself never needs it, since preserving arrival
	/// order already guarantees every admissible class eventually becomes
	/// the oldest pending work and gets picked.
	std::chrono::microseconds starvation_threshold{0};
};

	/// Arachne's index-agnostic operation scheduler: the layer between
/// Controller and IAdapter that turns a stream of individual Traverse/Modify
/// requests into batches, and runs those batches against the adapter on a
/// small pool of worker threads. It only ever sees TraverseRequest/
/// ModifyRequest -- no index or GPU knowledge lives here.
///
/// Three stages, each with its own thread(s):
///
///   caller thread(s)              planner thread              worker threads (0..N-1)
///   ---------------                --------------              -----------------------
///   schedule(request)              plannerLoop():              workerLoop(i):
///     enqueue TraverseTask/  -->     policy_->chooseBatchKind()   dequeue batch  <--
///     ModifyTask into queue_         collectBatch() via            from
///     (stage 1); return a            policy_->selectCandidateIndex/ dispatch_queue_
///     future                         canAppendToBatch()             (stage 3)
///                                    (stage 2)                     executeBatch():
///                                    push batch onto      -->        IAdapter::traverse|modify
///                                    dispatch_queue_                 Host|Device(batch)
///                                                                    on_complete(result) (Traverse)
///   future.get()  <-------------------------------------------------  promise.set_value(result)
///
/// Stage 2's actual ordering/grouping strategy is pluggable via
/// SchedulingPolicy (scheduling_policy.hpp); OpScheduler itself just drives
/// the three-stage pipeline and owns the queues/threads. traverse_batch_size_/
/// modify_batch_size_ cap how large a batch stage 2 builds before handing it
/// to stage 3; batch_wait_timeout_ optionally lets a non-empty batch wait a
/// bit for more eligible ops rather than dispatching immediately.
///
/// Stage 2 also runs a Traverse/Modify execution-admission gate, right after
/// SchedulingPolicy picks a batch_kind and before collectBatch() builds it
/// (see canAdmit()/reserveExecutionSlot()/releaseExecutionSlot()). This is
/// separate from SchedulingPolicy on purpose -- SchedulingPolicy only ever
/// looks at one batch-in-progress against queue_ (composition), never at
/// what's already executing on the worker pool (cross-batch admission), and
/// mixing the two concerns into one interface would make every policy
/// implementation re-solve the same admission problem. The gate itself is
/// deliberately consulted here, in the planner thread, rather than inside
/// workerLoop()/executeBatch(): a worker that can't yet run a batch it
/// already popped would sit blocked holding nothing useful, whereas a batch
/// the planner hasn't built yet simply stays represented as ordinary pending
/// ops in queue_ -- other, non-conflicting batches keep flowing to workers
/// meanwhile, and no worker thread is ever tied up waiting.
///
/// Rule enforced when an adapter's IAdapter::requiresTraverseModifyIsolation()
/// returns true (the default): Traverse batches may run concurrently with
/// other Traverse batches; a Modify batch may run concurrently with other
/// Modify batches of the *same* ModifyOp; a Modify batch of a *different*
/// ModifyOp, or any Modify running alongside any Traverse, must wait for the
/// conflicting batch(es) to finish first. An adapter returning false from
/// requiresTraverseModifyIsolation() opts out entirely -- the gate always
/// admits, identical to this feature not existing.
///
/// Choosing *which* (kind, op) class to build the next batch from is not
/// simply "ask SchedulingPolicy once and commit": that would let a single
/// inadmissible answer stall the whole pipeline (worker threads idle, other
/// perfectly runnable work stuck behind it in queue_) whenever
/// SchedulingPolicy's preferred kind happens to conflict with whatever's
/// currently in flight. Instead, each planner iteration:
///   1. Scans queue_ for every (kind, op) class that both has pending work
///      *and* currently passes canAdmit() (admissibleClasses()) -- at most
///      three: Traverse, Modify+Insert, Modify+Delete.
///   2. If none are admissible, the planner genuinely has nothing safe to
///      run and waits (the isolation rule above allows no alternative --
///      not a scheduling gap, a real floor).
///   3. Otherwise, SchedulingPolicy::chooseBatchKind() is still asked for
///      its preference and honored whenever that preference is itself
///      admissible -- SchedulingPolicy retains full authority over the
///      common case. Only when its preferred kind isn't currently
///      admissible does the planner fall back to whichever *other*
///      admissible class has waited longest, rather than blocking while
///      that work sits ready to run (selectNextBatchKind()).
///
/// Starvation override: step 3's fallback already keeps FifoSchedulingPolicy
/// starvation-free (arrival order means a class that's repeatedly bypassed
/// is, by construction, the oldest pending work, so it's exactly what the
/// fallback picks first). A different, pluggable SchedulingPolicy has no
/// such guarantee -- one that always prefers Traverse whenever any Traverse
/// is pending, say, would never let chooseBatchKind() return Modify at all,
/// so a Modify class could sit admissible-but-never-selected indefinitely
/// even though nothing is actually blocking it. SchedulingConfig::
/// starvation_threshold guards against exactly this: if the oldest pending
/// item of *any* admissible class has waited at least that long, the
/// planner forces that class to be scheduled next regardless of what
/// SchedulingPolicy would have preferred. 0 (the default) disables this --
/// matches the feature not existing, since FifoSchedulingPolicy never needs
/// it.
class OpScheduler {
 public:
  explicit OpScheduler(SchedulingConfig config = {},
                       std::unique_ptr<SchedulingPolicy> policy = nullptr);
  ~OpScheduler();

  OpScheduler(const OpScheduler&) = delete;
  OpScheduler& operator=(const OpScheduler&) = delete;

  /// Connects this scheduler to an adapter. Must be called before schedule().
  /// `on_worker_start`, if provided, runs exactly once per execution worker
  /// thread before it processes any batch, with that worker's 0-based index.
  /// OpScheduler never interprets it -- Controller uses it to bind each
  /// worker to its own CUDA stream without OpScheduler needing to know CUDA
  /// exists (see class doc comment).
  void start(IAdapter& adapter, std::function<void(std::size_t)> on_worker_start = nullptr,
				 std::function<void(TraverseRequest&)> prepare_traverse = nullptr,
				 std::function<void(ModifyRequest&)> prepare_modify = nullptr);

  /// Stops worker processing and drains queues before shutdown.
  void shutdown();

  /// Schedules one Traverse request. `on_complete`, if provided, runs on the
  /// worker thread that computed the result, right before the returned
  /// future is made ready -- guaranteed to have already run by the time
  /// future.get() unblocks (or, on failure, before future.get() rethrows).
  /// Fires exactly once regardless of outcome (see TraverseTask::on_complete's
  /// doc comment) -- lets a caller move result-dependent bookkeeping, or a
  /// chained follow-up dispatch, off of whichever (unboundedly many) thread
  /// calls future.get(), onto the bounded worker pool instead. OpScheduler
  /// never interprets this itself.
  std::future<TraverseResult> schedule(
      TraverseRequest request,
      std::function<void(std::exception_ptr, const TraverseResult&)> on_complete = nullptr);

  /// Schedules one Modify request. Returns a future that becomes ready once
  /// execution finishes. `on_complete` has the same contract as the
  /// TraverseRequest overload's.
  std::future<ModifyResult> schedule(
      ModifyRequest request, std::function<void(std::exception_ptr, const ModifyResult&)> on_complete = nullptr);

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
									 std::size_t batch_target, std::unique_lock<OpSchedulerMutex>& lock);
	// Dispatches a whole batch (kind- and mode-homogeneous, see
	// SchedulingPolicy::canAppendToBatch()) to exactly one IAdapter Host/Device
	// entry point in a single call, rather than looping one request at a time
	// -- see those methods' doc comments (adapter/index_adapter.hpp) for why.
	void executeBatch(ScheduledOperationBatch batch);
	void executeTraverseBatch(ScheduledOperationBatch batch);
	void executeModifyBatch(ScheduledOperationBatch batch);

	// Traverse/Modify execution-admission gate -- see class doc comment and
	// IAdapter::requiresTraverseModifyIsolation(). canAdmit()/
	// reserveExecutionSlot() assume the caller already holds mutex_ (only
	// ever called from plannerLoop(), which holds it for the whole loop
	// body); releaseExecutionSlot() takes mutex_ itself (called from
	// workerLoop(), after executeBatch() returns, outside any lock).
	bool canAdmit(ScheduledKind kind, std::optional<ModifyOp> op) const;
	void reserveExecutionSlot(ScheduledKind kind, std::optional<ModifyOp> op);
	void releaseExecutionSlot(ScheduledKind kind);

	// One (kind, op) execution-admission class with pending work in queue_ --
	// see class doc comment's "Choosing *which* (kind, op) class..."
	// paragraph and admissibleClasses()/selectNextBatchKind() below. `op` is
	// engaged only when `kind == ScheduledKind::Modify`.
	struct PendingClass {
		ScheduledKind kind;
		std::optional<ModifyOp> op;
		std::chrono::steady_clock::time_point oldest_enqueued_at;
	};

	// Scans queue_ for every (kind, op) class -- Traverse; Modify+Insert;
	// Modify+Delete, at most three -- that currently has at least one
	// pending op *and* passes canAdmit() right now, recording each present
	// class's oldest (longest-waiting) member. Assumes the caller already
	// holds mutex_, same as canAdmit()/collectBatch().
	std::vector<PendingClass> admissibleClasses() const;

	// Picks which class plannerLoop() should build its next batch from, out
	// of `admissible` (see admissibleClasses()) -- nullopt if `admissible`
	// is empty (nothing pending is currently safe to run at all). See class
	// doc comment for the full preference order (starvation override,
	// then SchedulingPolicy's own preference if admissible, then whichever
	// other admissible class has waited longest).
	std::optional<PendingClass> selectNextBatchKind(const std::vector<PendingClass>& admissible) const;

	// Removes and returns the single oldest queue_ entry matching (kind,
	// op) -- used to seed a new batch with exactly the class
	// selectNextBatchKind() chose, before collectBatch() fills the rest via
	// the normal SchedulingPolicy-driven path (canAppendToBatch()'s
	// existing mode/op-homogeneity check against the seeded entry keeps the
	// rest of the batch consistent with it). Only ever called immediately
	// after admissibleClasses()/selectNextBatchKind() confirmed a matching
	// entry exists, under the same uninterrupted mutex_ hold -- throws
	// std::logic_error if that invariant somehow doesn't hold.
	ScheduledOperation extractOldest(ScheduledKind kind, std::optional<ModifyOp> op);

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
	std::chrono::microseconds starvation_threshold_;

	// Strategy
	std::unique_ptr<SchedulingPolicy> policy_;

	// Set once in start(), before any worker thread is spawned -- safe to
	// read from workerLoop() without holding mutex_ (thread creation
	// establishes a happens-before edge for everything start() did first).
	std::function<void(std::size_t)> on_worker_start_;
	std::function<void(TraverseRequest&)> prepare_traverse_;
	std::function<void(ModifyRequest&)> prepare_modify_;

	// Lifecycle / threading
	mutable OpSchedulerMutex mutex_;
	OpSchedulerCondVar cv_incoming_;
	OpSchedulerCondVar cv_dispatch_;
	bool running_ = false;
	bool stop_requested_ = false;
	std::thread planner_;
	std::vector<std::thread> execution_workers_;

	// Pending work queues and ids
	ScheduledOperationQueue queue_;
	std::deque<ScheduledOperationBatch> dispatch_queue_;
	std::uint64_t next_id_ = 1;

	// Traverse/Modify execution-admission gate state -- guarded by mutex_,
	// see canAdmit()/reserveExecutionSlot()/releaseExecutionSlot(). Set once
	// in start() from adapter.requiresTraverseModifyIsolation(), before any
	// worker thread is spawned -- same happens-before reasoning as
	// on_worker_start_ above applies to reading it afterward.
	bool traverse_modify_isolation_enabled_ = true;
	std::size_t inflight_traverse_ = 0;
	std::size_t inflight_modify_ = 0;
	std::optional<ModifyOp> inflight_modify_op_;

	// Adapter call target (owned outside this class)
	IAdapter* adapter_ = nullptr;
};

}  // namespace arachne
