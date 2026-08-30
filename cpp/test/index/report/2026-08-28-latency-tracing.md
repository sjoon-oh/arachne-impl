# Latency tracing: where does the best/worst config's time actually go?

Status: **in progress**. Follow-up to
[2026-08-27-replacement-policy-sweep.md](2026-08-27-replacement-policy-sweep.md)'s
final ranking. That sweep only had aggregate counters
(`candidates_requeued`, `relocation_batches`, ...) and op-averaged
ms/op -- no measurement of where wall-clock time is actually spent. This
entry uses the existing `ARACHNE_ENABLE_TRACING` build (`build-trace/`,
already configured from an earlier session) to get real per-scope latency
for the best and worst configs identified in that sweep.

Raw trace CSVs referenced here live alongside this file, under
`2026-08-28-latency-tracing/` (same base name as this report, per the
project's convention for this kind of artifact).

## Correction to the prior entry's "beats raw" framing

`raw_hnswlib` runs single-threaded in every config of the prior sweep --
`RunRawHnswlib()` never calls `RunConcurrently`/`RunAsyncBatch` or touches
`--exec-threads`/`--client-threads` at all (verified: zero matches for
`std::thread`/`RunConcurrently`/`client_threads` in that function). So the
prior entry's "2 of 24 configs beat raw hnswlib" claim compared an
8x-parallel Arachne config against a fixed 1-thread raw baseline, not an
apples-to-apples comparison. The one truly fair comparison in that sweep --
`threads=1` against `threads=1` -- has Arachne 6-17x slower than raw even at
its best-tuned `batch` value there. This doesn't invalidate the sweep's
internal (Arachne-vs-Arachne) comparisons across configs, only the
raw-vs-Arachne "beats raw" framing specifically.

## What's being compared

From the prior sweep's final ranking (raw_hnswlib total: 797,169 ms for
900,000 ops):

- **Worst**: `batch=1, exec_threads=1, client_threads=1` -- every flag's own
  documented default, 11.47x raw's total time.
- **Best**: `batch=256, exec_threads=4, client_threads=4` -- 0.84x raw's
  total time (the only insert-beats-raw point in the whole sweep).

Both reuse the same saved 1M-vector graph
(`/tmp/arachne_cost_aware_1m_sweep/arachne_1m_index.bin`, from the prior
sweep's config 1 build) via `--load-index`, so build time is identical and
excluded from both runs -- this is purely about the streaming
insert/delete/search + Coordinator work.

## Instrumentation added (measurement-only, no behavior change)

Three of `region_manager.cpp`'s existing `ARACHNE_TRACE_SCOPE` call sites
were already present (`RegionManager::processRelocationBatch`,
`RegionManager::make`, plus `OpScheduler::executeTraverseBatch`/
`executeModifyBatch` and every `Controller::*` entry point). Added for this
investigation, to distinguish the three `candidates_requeued` code paths
discussed in conversation:

- `RegionManager::buildEvictionCandidates()` -- new scope,
  `"RegionManager","buildEvictionCandidates"`. This function's own doc
  comment already flagged it as `O(all tracked anchors)`; now measured
  directly instead of inferred.
- `RegionManager::buildRelocationPlan()`'s promotion-collection loop -- new
  scope, `"RegionManager","buildRelocationPlan_collect"`, wrapping exactly
  the `while (selectNextPromotionCandidate())` loop (requeue paths 1 and the
  budget-overflow case).
- `RegionManager::buildRelocationPlan()`'s victim-selection loop -- new
  scope, `"RegionManager","buildRelocationPlan_evict"`, wrapping exactly the
  `while (available + immediately_reclaimable < required)` loop.

No new statistics, no behavior change -- each addition is a brace-scoped
`ARACHNE_TRACE_SCOPE(...)` around code that already existed, compiled out
entirely unless `ARACHNE_ENABLE_TRACING` is on (see `telemetry/trace.hpp`'s
own file-level comment). Diff is in `cpp/src/core/region_manager.cpp`.

## Run setup

- `ARACHNE_TRACE_DIR` set to `/tmp/arachne_trace_worst` /
  `/tmp/arachne_trace_best` respectively -- both tmpfs-backed on this
  machine, chosen specifically to keep the tracing writes off real disk I/O
  per the investigation's own request to minimize measurement overhead.
- Same workload/budget/policy/group settings as the original sweep (1M
  base, 30 steps, 100 MiB budget, `cost_aware`,
  `group_merge_overlap_threshold=0.5`, `max_eviction_group_size=10`,
  `batch_wait_timeout_us=500`, seed=100), `--quiet-logs` to keep
  `ARACHNE_LOG_INFO`'s own overhead out of the measurement.
- Both launched together (background, separate `ARACHNE_TRACE_DIR`s so
  their CSVs don't collide) to keep total wall-clock investigation time
  down -- see the original sweep's report for why this matters (each 1M/30-
  step run takes on the order of an hour or more even with `--load-index`).

## Results

### Reproducibility caveat found first (important, read before trusting the rest)

Rerunning the exact "worst" config (`batch=1, threads=1`) via `--load-index`
gave dramatically different numbers than the original sweep's config 1 (which
did a real in-process build immediately before streaming):

| | original sweep (config 1, real build) | this trace run (`--load-index`) |
|---|---|---|
| insert ms/op | 15.7161 | 2.6767 |
| search ms/op | 13.1904 | 2.2901 |
| candidates_requeued | 1,609,806 | 640,680 |
| candidates_rejected | 12,911 | 70,656 |

~5.8x different on the *same* config at the *same* scale. The "best" config
(`batch=256, threads=4`) shifted too, but less (insert 1.5535->0.1702ms,
search 0.5232->0.7604ms). Net effect: the best/worst gap that looked like
~25x in the original sweep looks more like ~3x on this rerun. Leading
hypothesis: the original config 1 measurement happened immediately after
~28 minutes of real in-process graph construction (memory allocator/cache/
CPU-frequency-governor state warmed up differently) -- not yet confirmed.
**This means the original 24-config ranking's absolute numbers (and possibly
its ordering) should be treated as noisy, not as precise measurements.** The
qualitative conclusions this entry draws below are about *where* time goes
within one internally-consistent run, which doesn't depend on the
sweep's absolute ranking being exactly right.

### `RegionManager::recordTraversal` is the dominant single cost, but the
mean is misleading -- it's a heavy-tailed distribution, not a uniform
per-call tax

Aggregate view (matches what was reported in conversation): mean 2.21 ms/call
(worst) / 1.50 ms/call (best), ~600,000 calls each (once per insert+search),
totaling 85%+ of the whole run's insert+search+delete time.

Percentiles change the story completely -- see
`recordtraversal_latency.png` (per-call latency vs. call order, both
configs, raw scatter + rolling mean):

| | worst | best |
|---|---|---|
| p50 | 0.018 ms | 0.013 ms |
| p95 | 0.087 ms | 0.219 ms |
| p99 | 0.138 ms | 0.428 ms |
| mean | 2.207 ms | 1.499 ms |
| **max** | **7,003.7 ms** | **216,864.5 ms** |

The *typical* call is fast (tens of microseconds) and entirely reasonable.
The mean is dragged up by a small number of catastrophic outliers -- one
single `recordTraversal()` call took **216.9 seconds** in the best config's
run. This is not "too many anchors get inserted per call on average" (the
conversation's working hypothesis going in) -- it's a small number of calls
getting stuck for a very long time.

### Ruled out: `RegionManager::mutex_` wait time

`RegionManager::recordTraversal()`'s own lock (`RegionManagerMutex mutex_`,
already covered by the pre-existing `RegionManager-lockwait` trace scope)
cannot be the cause of those multi-second/multi-minute stalls -- its own
max observed wait is 6.7 ms (worst) / 20.6 ms (best), three to four orders
of magnitude too small.

### Hypothesis at the time (later confirmed by direct measurement --
see "Follow-up" below): contention on the separate
`CostAwareReplacementPolicy::mutex_`

`recordTraversal()` calls `replacement_policy_->onAnchorTouched(anchor_id)`
once per touched anchor, which takes `CostAwareReplacementPolicy`'s *own*,
separate `mutex_` (`replacement_policy.cpp:574-581`). That same mutex is
also taken by `evaluateAdmission()` (`replacement_policy.cpp:655`) for the
*entire duration* of its scan over `context.eviction_candidates` -- and
`evaluateAdmission()` runs once per candidate inside
`buildRelocationPlan_collect`'s loop, which (see the sweep report's earlier
finding) processed ~500,000 candidates across only 9 giant passes in the
best config, each pass averaging ~138 seconds. A worker thread's
`onAnchorTouched()` call landing during one of those giant passes could face
a long queue of frequent, individually-short lock re-acquisitions from the
Coordinator (a lock-convoy effect) rather than one long hold -- which would
explain both why per-acquisition RegionManager lockwait stays tiny *and*
why a `recordTraversal()` call can still stall for minutes. **Not yet
measured directly** -- there is no trace/instrumented-lock coverage on
`CostAwareReplacementPolicy::mutex_` today, so this is inference from
correlated timing (the worst outlier is far larger in the config whose
`buildRelocationPlan_collect` passes are also far longer), not a confirmed
mechanism. Adding a wait-time trace on that mutex is the natural next
measurement, not yet done.

### Artifacts in this directory

- `recordtraversal_latency.png` -- per-call latency vs. call order, both
  configs, generated from the raw trace CSVs via the (not-committed,
  throwaway) `/tmp/plot_recordtraversal.py`.
- `arachne_1m_index.bin` -- backup of the saved 1M-vector graph
  (`--save-index`/`--load-index` from the original sweep) both trace runs
  in this entry loaded via `--load-index`, so their starting state is
  reproducible. **277 MB, not yet added to git** -- flagged to the user
  rather than committed automatically.
- Full raw trace CSVs (multi-hundred-MB, `RegionManager-lockwait.csv` and
  `OpScheduler-lockwait.csv` alone are ~210-235 MB) were left in
  `/tmp/arachne_trace_worst/` and `/tmp/arachne_trace_best/` rather than
  copied here -- too large for a git-tracked report directory. Only the
  distilled PNG and the numbers quoted above are archived.

## Follow-up: instrumenting `CostAwareReplacementPolicy::mutex_` confirms the hypothesis

New measurement-only code (no behavior change): `CostAwareReplacementPolicy`'s
`mutex_` was swapped from plain `std::mutex` to the existing
`telemetry::InstrumentedMutex` (same drop-in wrapper `RegionManager::mutex_`
already uses), guarded by `#ifdef ARACHNE_ENABLE_TRACING` exactly like
`RegionManagerMutex` (see `core/replacement_policy.hpp`'s new
`CostAwareReplacementMutex` type and `core/replacement_policy.cpp`'s
`std::lock_guard<std::mutex>` -> `std::lock_guard` (CTAD) call sites, 49
of them, needed since the member's type is no longer always literally
`std::mutex`). Also added `ARACHNE_TRACE_SCOPE("CostAwareReplacementPolicy",
...)` to `onAnchorTouched()` (whole function) and to the locked sections of
`evaluateAdmission()` and `selectEvictionCandidate()`
(`evaluateAdmission_locked`/`selectEvictionCandidate_locked`). 356/357 tests
still pass on the normal (non-tracing) build (same pre-existing unrelated
failure as every other entry in this investigation) -- confirms the
`#else` branch (`using CostAwareReplacementMutex = std::mutex`) and the
CTAD lock_guard change are both behavior-neutral.

Two new runs at the *exact* configs requested: single-thread
(`batch=1, exec_threads=1, client_threads=1`) and a representative
multi-thread point (`batch=32, exec_threads=4, client_threads=4` -- not
one of the original sweep's extremes, chosen fresh for this comparison).
Both via `--load-index` against the same saved 1M graph.

| | single (batch=1, t=1) | multi (batch=32, t=4) |
|---|---|---|
| `CostAwareReplacementPolicy` mutex wait -- max | **6,561.99 ms** | **2,739.96 ms** |
| `CostAwareReplacementPolicy` mutex wait -- p50 / mean | 0.00002 / 0.0011 ms | 0.00003 / 0.0034 ms |
| `RegionManager` mutex wait -- max (comparison) | 20.98 ms | 34.77 ms |
| `buildRelocationPlan_collect` -- n calls / max | 21 / **701,621 ms (701.6 s)** | 8 / **593,458 ms (593.5 s)** |
| `evaluateAdmission_locked` -- n calls | 501,974 | 494,920 |
| `recordTraversal` -- max | 31,045.7 ms | 6,755.6 ms |

**Hypothesis confirmed by direct measurement.** `CostAwareReplacementPolicy::
mutex_`'s own wait-time distribution has the same shape as
`recordTraversal()`'s -- near-zero at the median (tens of microseconds),
but a multi-second tail -- while `RegionManager::mutex_`'s wait time stays
under 35 ms at its absolute worst in both runs. The mechanism, now directly
traced rather than inferred:

1. Both configs need to evaluate roughly the same ~500,000 admission
   decisions over the course of the run (workload-driven, not
   config-driven -- `evaluateAdmission_locked`'s call count is nearly
   identical, 501,974 vs. 494,920, despite wildly different batch/thread
   settings).
2. Those decisions are not spread evenly -- they land inside a handful of
   `buildRelocationPlan_collect` passes (21 for single, 8 for multi), and at
   least one such pass runs for **over 10 minutes straight** in both
   configs (701.6 s / 593.5 s).
3. During that one pass, the Coordinator thread calls `evaluateAdmission()`
   (and therefore locks/unlocks `CostAwareReplacementPolicy::mutex_`)
   hundreds of thousands of times in rapid succession -- each acquisition
   individually cheap (`evaluateAdmission_locked`'s own mean is
   0.12 ms / 0.07 ms), but the sheer re-acquisition rate creates a lock
   convoy.
4. Any worker thread's `recordTraversal() -> onAnchorTouched()` call landing
   during that window has to queue behind that convoy rather than one long
   hold, explaining both why `RegionManager::mutex_` itself stays
   uncontended (a completely different lock) and why
   `CostAwareReplacementPolicy::mutex_`'s wait time -- not its hold time --
   is what balloons.

This mechanism reproduced across three independent runs at three different
(batch, threads) configs (the original best/worst pair and this new
single/multi pair), each with different absolute numbers (see the
reproducibility caveat above -- run-to-run noise is real and substantial)
but the *same qualitative shape*: near-instant median, catastrophic tail,
tied to a `buildRelocationPlan_collect` pass running many minutes in a
single stretch. The consistency of the mechanism across otherwise-noisy
absolute numbers is the strongest evidence in this investigation for it
being real rather than an artifact of one run.

### What this changes about the earlier "possible improvement directions"

The original sweep report's suggestion to cap
`max_promotion_bytes_per_pass`/`max_eviction_bytes_per_pass` remains
relevant, but this measurement sharpens *why*: it's not primarily about
limiting how much GPU memory moves per pass, it's about limiting how long
the Coordinator can monopolize `CostAwareReplacementPolicy::mutex_` in one
uninterrupted stretch. A cap on *candidates evaluated per pass* (independent
of, or in addition to, a byte cap) would bound
`buildRelocationPlan_collect`'s own duration directly, which this
measurement shows is the actual lever -- not batch size or thread count,
which only changed how the same underlying ~500K-decision cost got
distributed across passes, not the cost itself.

No code fix has been applied -- this entry stops at measurement and
diagnosis, per the investigation's own scope (tracing/measurement code only,
no behavior changes) established earlier.

## `shouldYieldPass()` fix implemented and verified -- net result is NOT a clean win

Implementation (measurement phase is over -- this is a real behavior change,
approved and applied): `ReplacementPolicy::shouldYieldPass()` new virtual
hook (default `false`, fully backward-compatible for the 5 non-`cost_aware`
policies), `RelocationBatchContext` gained `candidates_examined_this_pass`/
`pass_elapsed`, `buildRelocationPlan()`'s collect loop asks the hook
*before* pulling each candidate (no requeue needed when it yields --
nothing has been popped yet). `CostAwareReplacementPolicy` gained
`max_pass_duration` (default 20ms) and implements the hook by comparing
`context.pass_elapsed` against it. 6 new unit tests added (4 pure-function
tests of `shouldYieldPass()` in `replacement_policy_test.cpp`, 1 integration
test in `region_manager_coordinator_test.cpp` proving the RegionManager-side
wiring splits one `waitIdle()` into multiple passes without ever needing to
requeue). Full suite: 362/363 (same pre-existing unrelated failure).

Reran the exact same single (`batch=1,threads=1`)/multi (`batch=32,threads=4`)
configs against the fixed build, same saved 1M graph via `--load-index`.

### The narrow goal was achieved

| | single, before fix | single, after fix | multi, before fix | multi, after fix |
|---|---|---|---|---|
| `CostAwareReplacementPolicy` mutex wait -- max | 6,561.99 ms | **1,721.66 ms** | 2,739.96 ms | 12,589.78 ms (worse) |
| `buildRelocationPlan_collect` -- max single pass | 701,621 ms | 1,164.76 ms | 593,458 ms | 4,135.88 ms |
| insert ms/op | 2.6767 | **2.4470** | 6.0616 | **2.1150** |
| search ms/op | 2.2901 | **2.0730** | 1.6774 | 2.1449 (worse) |

Pass duration itself is now genuinely bounded (no more 10+-minute single
passes) for both configs, and `single`'s worst-case mutex wait dropped
~3.8x. Per-operation insert/search cost improved for `single` and for
`multi`'s insert.

### But a much bigger, unanticipated cost showed up: `buildEvictionCandidates()`'s per-pass cache is now defeated

`RegionManager::buildEvictionCandidates()` -- an `O(all tracked anchors)`
scan, by its own doc comment -- was designed to be computed **once per
`buildRelocationPlan()` call** and reused for the rest of that same pass
(`eviction_candidates_cache`). Forcing far shorter passes multiplies how
often a *new* pass (and therefore a fresh, uncached scan) is needed:

| | single | multi |
|---|---|---|
| `relocation_batches_total` (committed passes) | 21 -> **54,870** | 8 -> **3,986** |
| `processRelocationBatch` calls (incl. empty passes) | -- | 399,678 (single) |
| `buildEvictionCandidates` calls | -- | **399,669** (single) / 3,945 (multi) |
| `buildEvictionCandidates` total time | -- | **11,682 s (3.25 hours!)** (single) / 14.5 s (multi) |
| **whole-run wall clock** | ~1.3-1.9h range (this investigation's typical) | **4h13m+** (single) |
| sum of insert+delete+search timers | -- | 1,442 s (~24 min) |

For `single`, the gap between the *timed* insert/search/delete work
(~24 min total) and the *actual* wall-clock run time (4h13m+) is almost
entirely `buildEvictionCandidates()` being recomputed from scratch on
essentially every one of ~400K passes instead of a handful -- 3.25 hours of
pure redundant O(N) scanning that didn't exist before this fix. `multi` is
far less affected (3,945 calls, 14.5s total) because its higher batch size
and thread count mean the Coordinator still gets larger batches of pending
work per wakeup even with the 20ms cap, so the pass count only grew ~500x
instead of ~2,600x.

### Net verdict

**Not a clean win as shipped (20ms default).** The mechanism does what it
was designed to do (bound worst-case pass duration, and for `single`, bound
worst-case `CostAwareReplacementPolicy::mutex_` wait too), but at
`traverse_batch_size=1`/low-throughput settings the resulting pass-count
explosion defeats an existing, load-bearing cache and makes total wall-clock
dramatically worse -- worse than the problem it fixed, for that specific
config. `multi`'s regression is smaller but real too: worst-case mutex wait
actually got *worse* (2.7s -> 12.6s), and search regressed 27% -- bounding
pass *duration* alone doesn't guarantee a starved thread's wait improves
end-to-end if it just loses the race more often across more, shorter
passes instead of once across one long one.

### Options going forward (none implemented -- for discussion)

1. **Raise `max_pass_duration`'s default substantially** (e.g. 200-500ms
   instead of 20ms) -- fewer, larger passes reduces the
   `buildEvictionCandidates()` cache-defeat effect while still eliminating
   the original 700-second-class outliers.
2. **Decouple `buildEvictionCandidates()`'s cache from per-pass scope** --
   cache it with a time-based or state-based TTL shared across multiple
   consecutive passes (invalidated when a promotion/eviction actually
   commits, not on every `buildRelocationPlan()` call), so short passes
   stop each paying the full O(N) scan cost independently. Likely the more
   principled fix, but a larger change to a function three separate parts
   of this investigation now depend on.
3. **Add a minimum-candidates-examined floor alongside the duration cap**
   (`shouldYieldPass()` only yields if *both* `pass_elapsed >=
   max_pass_duration` *and* `candidates_examined_this_pass >= some floor`)
   -- prevents degenerate near-single-candidate passes without giving up
   the duration bound entirely.
4. **Revert `shouldYieldPass()`** and treat the original finding (a
   single pass can run for 10+ minutes) as accepted, now-understood
   behavior rather than something to fix -- given how large the mutex-wait
   improvement was for `single` specifically (6.6s -> 1.7s max), this may
   be premature, but is the safe baseline to fall back to.

Not decided here -- surfaced to the user for direction rather than picking
one unilaterally, given the mixed/regressed results above.

## Final verification: `touch_queue_` redesign implemented, tested, and confirmed to fix the original problem

Implementation (approved, real behavior change): `CostAwareReplacementPolicy::
onAnchorTouched()` no longer touches `mutex_` at all -- it pushes to a new,
separately-locked `touch_queue_`/`touch_queue_mutex_`. A new private
`drainTouchQueueLocked()` applies queued touches to `resident_`'s heat,
called (always already under `mutex_`, so no new lock acquisition on the
Coordinator side) from the top of every Coordinator-thread method that
reads `resident_`: `evaluateAdmission()`, `selectEvictionCandidate()`,
`selectNextEvictionCandidate()`. `shouldYieldPass()`'s default
`max_pass_duration` reverted to `0` (unlimited/disabled) -- the mechanism
stays available and tested, just off by default, since the root cause it
was compensating for is what this redesign actually fixes.

9 new/updated unit tests (2 new: a concurrency test proving
`onAnchorTouched()` never blocks behind a deliberately-long
`evaluateAdmission()` scan, and a drain-correctness test; 1 updated to match
`max_pass_duration`'s new default). Full suite: **364/365** (same
pre-existing unrelated failure as every entry in this investigation).

### Operational note: this verification pass hit a real disk-space incident

Mid-run, `/tmp` (which turned out to be real disk on the root partition, not
tmpfs as assumed when choosing it for "minimize overhead" earlier in this
investigation) filled to 99% from accumulated trace directories across this
investigation's many rounds, to the point where even the harness's own
tooling could not write. Recovered by deleting already-analyzed trace
directories (their key numbers were already captured in this report) via a
tool that didn't hit the same failure, freeing ~10.5 GB. One casualty: the
first `single_v2` run's trace CSVs never flushed (`TraceCollector` only
writes at process-exit, so the run itself was unaffected and its printed
summary survived, but the fine-grained CSVs were lost) -- re-run once space
was freed (only ~22 minutes this time, see below) to recover them. Lesson
for any future round of this investigation: clean up old trace directories
*before* starting a new one, not after -- `ARACHNE_TRACE_DIR` should point
somewhere with headroom tracked explicitly, not assumed free.

### Results: the original problem is fixed

| | single (batch=1,t=1) | multi (batch=32,t=4) |
|---|---|---|
| `RegionManager::recordTraversal` -- max | 216,864.5 ms (first measurement) / 7,003.7 ms / 31,045.7 ms / 6,755.6 ms (various earlier attempts) | (see multi row below) |
| `RegionManager::recordTraversal` -- max, **this fix** | **14.16 ms** | **7.45 ms** |
| insert ms/op (raw: 1.8885) | **0.4721** (4x faster than raw) | **0.1842** (10x faster than raw) |
| search ms/op (raw: 0.7682) | **0.0449** (17x faster than raw) | **0.0156** (49x faster than raw) |
| whole-run wall clock | **~22 min** (was 4h13m+ with the `shouldYieldPass`-only fix, ~1.3-1.9h before any fix) | (fast; exact figure not separately timed) |

`recordTraversal`'s worst case dropped from as bad as 216.9 seconds down to
**single-digit-to-low-double-digit milliseconds** -- essentially the noise
floor for a function doing real hash-map work. `single`'s total wall-clock
time also dropped from measured-in-hours back to ~22 minutes, confirming
the `buildEvictionCandidates()` cache-defeat side effect from the
`shouldYieldPass`-only attempt is gone now that passes aren't artificially
fragmented (`buildRelocationPlan_collect` n=2,233 for `single` here, vs.
54,870 committed passes when `shouldYieldPass` was forcing splits, vs. 21
in the very first, unfixed measurement -- back in the same order of
magnitude as the original, unsplit behavior, now *without* the multi-minute
tail that behavior used to carry).

### New (smaller) finding: `onAnchorEvicted()` (the delete path) has the
same structural vulnerability, now the dominant remaining tail

`RegionManager::releaseAnchor()` (which calls
`CostAwareReplacementPolicy::onAnchorEvicted()`) is called for **every**
delete, not just ones that turn out to reference a tracked anchor -- and
unlike `onAnchorTouched()`, it's a correctness-sensitive, must-stay-
synchronous call (a deleted anchor must stop being considered for
promotion/eviction immediately, not on some later lazy drain). It still
locks `mutex_` directly, so it's exposed to exactly the same lock-convoy
pattern `onAnchorTouched()` used to be:

| | single | multi |
|---|---|---|
| `RegionManager::releaseAnchor` -- max | 21,521.52 ms | (not separately isolated, but `CostAwareReplacementPolicy-lockwait`'s max of 55,312.75 ms matches no other traced scope as closely) |
| `CostAwareReplacementPolicy-lockwait` -- max | 21,517.67 ms (matches releaseAnchor almost exactly) | 55,312.75 ms |
| delete ms/op (raw: ~0.0004-0.0005) | 1.8985 (elevated) | 2.6847 (elevated) |

This is a real, if much smaller-scale, echo of the original problem --
`anchor_evictions`/delete volume (thousands to ~300K calls, all paying the
mutex acquisition even when it's a no-op) is far lower than the
600,000-call touch volume that dominated before, so its effect on the
aggregate delete ms/op is real but nowhere near as catastrophic as
`recordTraversal`'s used to be. Not fixed here -- flagged as a natural
follow-up. Harder than the touch fix: `onAnchorEvicted()` can't become a
purely lossy/lazy hint the way a heat touch could (correctness, not just
staleness, is at stake), so a proper fix would need something like a fast,
separately-locked "pending eviction" marker that `selectNextPromotionCandidate()`/
`evaluateAdmission()` check and skip, with the *full* `resident_`/
`pending_candidates_` cleanup still deferred to the next Coordinator-side
drain -- more design work than the touch queue needed, not attempted in
this session.

### Conclusion

The user's originally-diagnosed problem (`RegionManager::recordTraversal()`'s
catastrophic tail latency, traced to `CostAwareReplacementPolicy::mutex_`
contention between the worker-thread touch signal and the Coordinator's
admission/eviction scans) is fixed and verified by direct before/after
measurement across two independent configs. A smaller, structurally
identical issue on the delete path was found as a side effect of fixing the
first one being visible now that it no longer dominates -- left for a
follow-up decision rather than implemented in this session.

### Final verification trace locations (not copied into git, same reasoning as earlier entries -- too large)

- `/tmp/arachne_trace_single_v2/` (903 MB) -- single (batch=1,threads=1), full CSVs, recovered via the re-run above.
- `/tmp/arachne_trace_multi_v2/` (707 MB) -- multi (batch=32,threads=4), full CSVs.
