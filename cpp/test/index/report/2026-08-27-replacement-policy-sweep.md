# Replacement-policy bottleneck sweep

Status: **calibration + 24-config `cost_aware` 1M-scale sweep complete**.
See "Full sweep complete (24/24 configs)" below for the final ranking and
synthesis. The 5 non-`cost_aware` policies were scoped out of the 1M-scale
matrix after calibration (see "Findings so far" below) and remain
untested at that scale -- a listed follow-up, not done here. See
[2026-08-27-group-eviction.md](2026-08-27-group-eviction.md) for the prior
episode this continues from.

## Motivation

The group-eviction fix (previous entry) verifiably solved the bottleneck it
targeted, but didn't move end-to-end throughput, because a second bottleneck
(hysteresis-based admission in `CostAwareReplacementPolicy`) was hiding
behind the first and is now dominant. Open question: is *this* second
bottleneck specific to `CostAwareReplacementPolicy`'s own admission logic,
or would any policy hit a comparable wall under the same budget pressure --
i.e., is the real bottleneck somewhere shared (`OpScheduler`, `RegionManager`
locking, per-request overhead) rather than in the cost-aware scoring math
specifically?

Arachne ships 6 `ReplacementPolicy` implementations
(`cpp/include/core/replacement_policy.hpp`): `fifo`, `lru`, `lfu`, `clock`,
`twoq`, `cost_aware`. Comparing all 6 under identical traffic/budget
isolates this.

## Structural difference found before running anything

`ReplacementPolicy::evaluateAdmission()`'s base-class default is
**unconditional Admit**. `fifo`/`lru`/`lfu`/`clock`/`twoq` do not override
it -- only `cost_aware` does. So the 5 non-cost-aware policies have **no
capacity-aware rejection at all**: if a promotion candidate needs space and
*any* victim is selectable (however unfavorable), it's evicted and the
candidate promoted, no cost/benefit comparison. Consequence found by smoke
test (see below): under a budget too small to hold the working set, these
policies don't reject-and-move-on the way `cost_aware` does -- they run a
full evict+promote cycle for essentially every touched anchor, continuously.
This is a real, structural difference in how each policy responds to
sustained budget pressure, not a bug -- but it means per-policy runtimes
under the same tight budget may differ by orders of magnitude, which changes
how this sweep has to be paced (see "Revised approach" below).

## Test infra changes (test-only, no core library changes)

`cpp/test/index/hnsw_workload_compare.cpp`:

- `--replacement-policy {cost_aware,fifo,lru,lfu,clock,twoq}` -- Controller's
  `replacement_policy` ctor argument was already pluggable
  (`std::unique_ptr<ReplacementPolicy>`), just never exposed on this binary
  before now. `MakeReplacementPolicy()` factory added.
- `--save-index PATH` / `--load-index PATH` -- wraps `HnswlibIndex::exportTo()`/
  `loadFrom()` (already existed, wrapping hnswlib's own `saveIndex()`/
  `loadIndex()`). Motivation: `RunArachneController()`'s `index.build(...)`
  call happens *before* the `Controller` is even constructed -- build is
  completely independent of policy/batch-size/thread-count, so a sweep that
  rebuilds the same graph for every config wastes the dominant cost
  (build time) on identical, deterministic work. Build once, save, `--load-index`
  for every subsequent config in the sweep.
- Smoke-tested both flags at small scale (5000 base / 2 steps): save
  produces a working index file, load completes in ~5ms vs ~4.5s for a real
  build at that tiny scale (this ratio should widen further, in Arachne's
  favor, at 1M scale).

## Sweep plan (as agreed with the user)

Fixed across all configs:
- Workload: 1M base vectors, 30 steps (matches the earlier "100K wasn't
  large enough to see a clear signal" finding).
- GPU budget: 30% of full residency footprint, same ratio as every prior
  run in this investigation. Scaled proportionally from the established
  100K-scale number (10 MiB was ~30% of ~35MB estimated full residency) ->
  **100 MiB (104,857,600 bytes)** at 1M scale.
- `group_merge_overlap_threshold=0.5`, `max_eviction_group_size=10` for all
  6 policies (already validated safe/beneficial; keeping it fixed isolates
  policy x batch-size x thread-count effects instead of conflating them with
  group-eviction on/off).
- `batch_wait_timeout_us=500` (batch_size only bounds *already-queued*
  requests without a nonzero wait -- see the binary's own flag doc comment).
- `raw_hnswlib` baseline run **once** at this scale (build/policy/batch are
  irrelevant to it), reused as the fixed reference for every row rather than
  re-run per config.

Swept:
- `traverse_batch_size = modify_batch_size` in (1, 4, 8, 16, 32, 64, 128, 256).
- `(exec_threads, client_threads)` pairs in (1,1), (4,4), (8,8) -- paired
  rather than independent, per the user's explicit direction (worker-thread
  count and submitting-client count matched, considering the Coordinator's
  own background thread already competing for CPU). Note this leaves
  `--exec-threads` above 1 for the first time in this investigation --
  the file's own comment documents `exec_threads=1` as a deliberate
  determinism choice (hnswlib's per-node locking makes concurrent
  `addPoint()` interleaving, and therefore recall@k, depend on OS thread
  scheduling). Accepted here because recall@k is *already* invalid at this
  scale regardless (`--limit-base 1000000` is still short of the workload's
  full 10M base pool the groundtruth was computed against), so there is
  nothing left to lose in this specific sweep.
- 6 replacement policies.

6 x 8 x 3 = **144 arachne-only runs**, plus 1 shared raw baseline and 1
shared arachne build+save pass.

## Revised approach after the fifo smoke test

Initial time estimate (~22-24h for a 48-run matrix, before the user added
the exec/client-thread pairing dimension) was based on `cost_aware`'s
observed per-run cost, which includes its rejection path short-circuiting
most budget-pressure situations quickly. The structural difference above
(non-cost_aware policies never reject, so they run full churn instead) means
that estimate cannot be assumed to hold for the other 5 policies without
checking first -- a policy that thrashes continuously under a 30%-of-full
budget could take dramatically longer than `cost_aware` does at the same
config, and a 144-run matrix is too expensive to discover that the hard way
partway through.

Plan: run one calibration pass per policy at a smaller, fast-to-complete
scale (matching this investigation's existing 100K/15-step calibration
point) before committing CPU-hours to the full 1M/30-step/144-config matrix,
specifically to get a real per-policy wall-clock number under the *same*
30%-budget pressure ratio this sweep intends to use throughout. Results and
the resulting (possibly revised) full-matrix time estimate will be appended
below once that calibration pass finishes.

## Findings so far

### `fifo` under tight budget: does not complete in reasonable time, likely
unbounded scaling, not just "slow"

Calibration smoke test: `--limit-base 5000 --limit-steps 2
--gpu-budget-bytes 2097152 --replacement-policy fifo` (5000 base + 20,000
insert = 25,000 total vectors ever touched, 2MiB budget -> ~5 resident
Regions at `vectors_per_region=1024`). This is a *tiny* workload deliberately
chosen to be fast -- the planned sweep's smallest calibration point.

Observed: the streaming step loop itself finishes quickly (`step 1/2 done`,
`step 2/2 done` both print within seconds), but the run never reaches its
final summary printout. `gdb -p <pid> -batch -ex "thread apply all bt"`
shows the main thread correctly parked in `Controller::waitIdle()` (working
as designed), while the `RegionManager` Coordinator thread is pegged at
99.9% CPU inside:

```
RegionManager::coordinatorLoop()
 -> RegionManager::processRelocationBatch()
  -> RegionManager::buildRelocationPlan()
   -> FifoReplacementPolicy::selectEvictionCandidate()
    -> ContainsCandidate()  (replacement_policy.cpp:26, std::any_of scan)
```

Confirmed twice independently (killed and re-ran) -- both times still
spinning at 99.9% CPU on one thread past the 2-minute mark, no progress
visible in the log after the step loop ends. This is **not a deadlock** (no
thread is blocked waiting on a lock/condvar that's never signaled) -- it's
a genuine busy loop making forward progress too slowly to matter.

**Suspected mechanism** (not yet fully isolated to a single root cause --
see "Open questions" below): `FifoReplacementPolicy::selectEvictionCandidate()`
does a linear scan of its whole `promoted_order_` list per call
(`replacement_policy.cpp:93-100`), called once per victim needed inside
`buildRelocationPlan()`'s victim-selection `while` loop
(`region_manager.cpp:915-971`). Because the base `ReplacementPolicy::
evaluateAdmission()` default is unconditional Admit (see "Structural
difference found before running anything" above), `fifo` never
rejects-and-moves-on the way `cost_aware` does under budget pressure -- every
touched anchor that needs to move gets a full evict-then-promote cycle. That
plausibly keeps `promoted_order_` large (bounded by how many distinct
anchors are concurrently resident-or-recently-evicted, not by the tiny
5-region budget) for the whole run, so the linear scan cost compounds across
many thousands of promotion attempts over the run's lifetime -- effectively
quadratic total work with a large constant, not strictly infinite, but slow
enough that "eventually finishes" was not observed within the time budget
of a smoke test that should complete in well under a minute.

### Consequence for the planned sweep

The 144-config, 1M-base/30-step matrix as planned assumed per-config runtime
in the same ballpark as `cost_aware`'s (~25 min), because that estimate was
built from `cost_aware`'s own behavior, which short-circuits most
budget-pressure situations via outright rejection. That assumption does not
hold for the 5 non-`cost_aware` policies -- confirmed here on a workload
1M/30-steps: **200x smaller** than the planned scale, under the same 30%
budget-pressure ratio. Running the full matrix as specified risks getting
stuck on the very first non-`cost_aware` leg for an unknown, possibly very
long time, with no clear signal from the driver script that it's "still
working" vs. "will never finish."

Paused here rather than continuing to guess or burn compute -- see the
conversation for the follow-up decision on how to proceed (raise the
per-policy budget so non-`cost_aware` policies aren't compared under
literally-unbounded thrash; fix the O(n) linear scan as a real
`ReplacementPolicy` performance issue in its own right; or scope the 5
non-`cost_aware` legs out of this particular sweep and treat "does it even
survive tight budget pressure" as itself part of the finding).

### Correction: it is not eviction-driven, and not a strict infinite loop

Follow-up control test: reran `fifo` at the identical tiny scale (5000
base / 2 steps) but with a **generous** budget (52 MiB -- confirmed via
`gpu_bytes_allocated=6703128` in the completed runs below that the real
working set is only ~6.4 MiB, so this budget should never require a single
eviction). With full `ARACHNE_LOG_INFO` logging (not `--quiet-logs`), the
string `"entering victim-selection loop"` (the line that only prints when
`buildRelocationPlan()` actually needs to evict something) **never appears
once** in 67,000+ log lines -- confirming eviction is never entered at all
under this budget. The run still did not finish within the 45s window
checked. So the earlier "stuck in `FifoReplacementPolicy::
selectEvictionCandidate()`'s O(n) scan under budget pressure" explanation
(from the first hang observation, at a *tight* 2 MiB budget) was at best
incomplete -- it cannot be the dominant mechanism, since the same slowness
shows up with essentially unlimited headroom. Also not a strict infinite
loop: distinct anchor ids climbing steadily through the log (up to 15,000+
of 25,000 total) confirm real forward progress is being made, just very
slowly.

### Real mechanism: promotion-candidate volume, not eviction

Ran all 5 non-`cost_aware` policies to completion at this same tiny
calibration scale (5000 base / 2 steps / 52 MiB budget, run in parallel,
`--quiet-logs`, each bounded by an 8-minute timeout as a safety net):

| policy | insert ms/op | search ms/op | regions_promoted | regions_evicted | candidates_rejected | relocation_batches | candidates_requeued |
|---|---|---|---|---|---|---|---|
| fifo  | 4.2626 | 4.3868  | 25 | 0 | 0 | 22   | 265  |
| lru   | 4.5257 | 5.6333  | 25 | 0 | 0 | 36   | 1424 |
| clock | 4.2442 | 5.9903  | 25 | 0 | 0 | 29   | 2248 |
| twoq  | 4.6484 | 7.5445  | 25 | 0 | 0 | 126  | 546  |
| lfu   | 6.2772 | 11.1612 | 25 | 0 | 0 | 1072 | 256  |

All 5 completed (no timeout needed), all show **zero evictions and zero
rejections** -- exactly as expected for policies whose `evaluateAdmission()`
never says no, under a budget that never runs out. `regions_promoted=25`
matches "the whole working set fits, gets promoted once, stays resident" --
i.e. there is no thrashing of any kind here. And yet every single one is
already far slower per-op than `cost_aware` has been at *any* point earlier
in this whole investigation (recall: `cost_aware` at 100K-scale/10MiB-budget,
i.e. a workload 20x larger under *real* budget pressure, still only ran
~3.05 ms/op insert and ~0.99 ms/op search -- see the previous report entry's
table). At this tiny, pressure-free scale, `search` in particular is 4.4x
to 11.3x worse across the 5 policies than `cost_aware` was at a *harder*
20x-larger-scale problem.

So the dominant cost for these policies is not eviction thrashing at all --
it is the sheer **volume of promotion-candidate processing**. Because
`evaluateAdmission()`'s default is unconditional Admit, every touched Region
on every traversal runs the full enqueue -> select -> admission-check ->
`RegionManager::make()` pipeline (Coordinator round-trip, locking,
bookkeeping) even when the outcome is a no-op "already resident, nothing to
do." `cost_aware` short-circuits most of this via outright rejection before
ever reaching `make()`; the other 5 never get that shortcut, so they pay
the full pipeline cost for essentially every touch. `lfu`'s outlier badness
(1072 relocation batches vs. 22-126 for the others, 6.3/11.2 ms/op) likely
compounds this with its own frequency-bucket bookkeeping (a `std::map` touch
plus list splice on every `onAnchorTouched()`), but that is not yet
confirmed by direct profiling, just relative-magnitude reasoning.

Control run: `cost_aware` at this exact same 5000-base/2-step/52MiB config,
run in isolation (no contention from the other 5):

| policy | insert ms/op | search ms/op | relocation_batches | candidates_requeued |
|---|---|---|---|---|
| cost_aware | 3.3849 | 4.8751 | 39 | 6157 |

Also zero evictions, zero rejections (same working set fits easily). This
is the surprising part: `cost_aware`'s search here (4.88 ms/op) is **worse**
than `fifo`'s (4.39 ms/op) and only roughly comparable to `lru`/`clock`
(5.63/5.99 ms/op) at this scale -- despite `cost_aware` being the one policy
with a real rejection shortcut. And `candidates_requeued=6157` is the
*highest* of all 6 policies at this scale (`lfu` next at 256, everything
else under 2250). The reject-shortcut cost_aware is expected to benefit
from clearly isn't paying off here, because nothing is being rejected --
there's no pressure to reject under a 52 MiB budget with only ~6.4 MiB of
real working set.

More striking: this ~4.9 ms/op search at 5000-base/generous-budget is
*worse* than `cost_aware` achieved at 100K-base/**10 MiB budget** (real,
sustained eviction pressure, 20x more data) in the previous report entry --
0.99 ms/op there. A smaller, pressure-free workload should not run slower
per-op than a larger, pressure-heavy one, if per-op cost were the only thing
varying. That strongly suggests a **large fixed cost per run** (independent
of op count) that this tiny 20,000-op calibration scale cannot amortize
away, but the 150,000-op scale used throughout the rest of this
investigation does -- e.g. Coordinator/`waitIdle()` drain overhead, CUDA
context/allocation warm-up, or similar one-time costs baked into every run
regardless of workload size.

### Conclusion for the calibration approach

This tiny-scale calibration point is good for one thing -- confirming
correctness and rough relative ordering under zero pressure (`fifo` best,
`lfu` worst among the 5; `cost_aware` unremarkable when nothing gets
rejected) -- but its *absolute* per-op numbers are not a reliable predictor
of behavior at the 1M-scale, real-pressure scale this investigation actually
cares about, because a fixed per-run cost dominates at this size. The
original sweep plan (`cost_aware` only, full 1M/30-step/24-config matrix,
build-once-reuse optimization) remains the right way to get a real signal --
see the conversation for the decision on whether/how to also get a real
(not fixed-cost-dominated) large-scale data point for at least one
non-`cost_aware` policy before ruling all 5 out of the full-scale
comparison entirely.

## Main-scale sweep: 1M base / 30 steps / 100 MiB budget, `cost_aware` only

Running (`sweep_cost_aware_1m.sh`, 24 configs: batch size in
{1,4,8,16,32,64,128,256} x (exec_threads, client_threads) in
{(1,1),(4,4),(8,8)}). `raw_hnswlib` and the arachne build/save both ran once,
paired in config 1 (`--engine both`); every other config loads the saved
1M-vector graph instead of rebuilding.

### Config 1 (`b1_t1`, batch=1, exec/client threads=1) -- the serial baseline

Total wall time for this one config: **4h07m52s** (config 1 alone paid the
one-time ~28.6-minute build for *both* engines; every later config skips
that via `--load-index`).

| | raw_hnswlib | arachne_controller | ratio |
|---|---|---|---|
| build | 1,716,809 ms | 1,712,981 ms | 1.00x (same-process, as expected -- build is policy/Controller-independent) |
| insert (300,000 ops) | 1.8885 ms/op | 15.7161 ms/op | **8.3x worse** |
| delete (300,000 ops) | 0.0005 ms/op | 1.5683 ms/op | 3,100x worse (both are tiny absolute numbers) |
| stream search (300,000 ops) | 0.7682 ms/op | 13.1904 ms/op | **17.2x worse** |
| mean recall@k (3 checkpoints) | 0.8186 | 0.8186 | identical (expected -- same graph, same query set) |

`controller stats: gpu_bytes_allocated=73,453,384 (73.5 MiB of the 100 MiB
budget) regions_promoted=5,690 regions_evicted=5,422 anchor_evictions=6,333
compactions=0 relocation_batches=19 candidates_requeued=1,609,806
candidates_rejected=12,911`.

**This gap is dramatically worse than every smaller-scale measurement
earlier in this investigation.** At 100K-base/10MiB-budget (previous report
entry), `cost_aware` ran ~3.05 ms/op insert and ~0.99 ms/op search -- roughly
2x and 1.5x worse than raw at that scale. At 1M-base/100MiB-budget (10x the
data, same 30%-of-full-residency budget ratio), the gap widened to 8.3x and
17.2x. The overhead is not a fixed per-op tax; it gets substantially worse
as the dataset grows.

### Root cause candidate: unbounded per-pass batch size

`relocation_batches=19` for a run that processed 600,000 traffic ops (300K
insert + 300K search) is suspiciously low -- at 100K-scale/150K-ops, this
same counter was in the 200-260 range (see the previous report entry's
table), i.e. *more* batches for *less* traffic. Combined with
`candidates_requeued=1,609,806` (over 84,700 requeues *per batch* on
average) and the multi-hour tail this investigation watched directly (see
below), the numbers point at one place:

`CoordinatorConfig::max_promotion_bytes_per_pass` and
`::max_eviction_bytes_per_pass` (`region_manager.hpp:202-203`) **default to
0, meaning unlimited**, and neither this test harness nor
`sweep_cost_aware_1m.sh` overrides them. `buildRelocationPlan()`'s
promotion-collection loop (`region_manager.cpp:810-891`) has no other cap on
how many candidates one pass absorbs into a single `plan.promotions` besides
the *entire* GPU budget being exhausted -- at 1M scale, with a 100 MiB
budget capable of holding thousands of Regions, a single Coordinator pass
can apparently pull in and process an enormous number of candidates before
ever finalizing a batch. Directly observed while this sweep's config 1 was
running: after all 30 streaming steps finished printing (i.e. all traffic
had already been submitted), the process sat in
`Controller::waitIdle() -> RegionManager::coordinatorLoop()` for **an
additional, unmeasured but clearly multi-hour stretch** -- `top -H`
confirmed the Coordinator thread pinned at 99.9% CPU with ~169 minutes of
accumulated CPU time at one mid-drain check alone, and the process's total
elapsed time (4h07m52s) is far larger than the sum of the two builds
(~57 min) plus what 600K ops at even the *worse* 100K-scale per-op rates
would predict (600K x ~3ms ~= 30 min). The unaccounted-for time is
consistent with one or a few enormous, slow-to-process final passes, not
per-op cost.

**This is a real, evidence-backed candidate, not yet a proven root cause.**
It has not been confirmed by direct profiling of a single
`buildRelocationPlan()` pass's candidate count at 1M scale (the same kind of
verification this investigation applied to the group-eviction and hysteresis
findings earlier) -- that would be the natural next step before touching any
code.

### Possible improvement directions (not yet implemented -- for discussion)

1. **Set `max_promotion_bytes_per_pass`/`max_eviction_bytes_per_pass` (or the
   fractional `*_fraction` convenience field) to a bounded value** and re-run
   this same config. Cheapest to try -- a config-only change, no core code
   touched. If `relocation_batches` goes up and per-op cost comes down, this
   confirms the unbounded-pass hypothesis directly. Risk: too small a cap
   could increase Coordinator wake-up overhead (more, smaller passes) or
   change eviction dynamics (a group's members could get split across passes
   -- worth checking `buildRelocationPlan()`'s "always allow the *first*
   candidate through even over cap" carve-out interacts correctly with the
   already-existing group-eviction code from the previous report entry).
2. **Profile one `buildRelocationPlan()` pass directly** (candidate count,
   wall time, lock-held time) at 1M scale before changing any config, to
   confirm which part of the pass is actually expensive -- the promotion-
   collection loop's `promotionBytes()` recomputation, the eviction
   candidate-building scan (`buildEvictionCandidates()`, documented as
   scanning every tracked anchor in its own comment), or something else
   entirely. This investigation's own track record (the "hysteresis" and
   "no eligible victim" findings) shows guessing the mechanism from
   aggregate numbers alone has been wrong before without a direct
   measurement to confirm it.
3. **Re-examine `trigger_interval` (100ms default) at this scale** -- if
   candidates are arriving far faster than every 100ms during the insert-
   heavy portions of a step, the Coordinator may rarely get a chance to
   finalize a *small* batch before more work has piled up, compounding the
   unbounded-pass effect above rather than being an independent cause.
4. **Compare against the trace-enabled build** (`ARACHNE_ENABLE_TRACING`,
   see the file overview) for a real breakdown of where wall time actually
   goes inside one pass, rather than inferring from aggregate counters --
   this investigation already has that build configured from earlier
   sessions (`build-trace/`).

None of these have been tried yet -- this section records the leads, not
conclusions. The sweep continues to run in the background across the
remaining batch-size/thread-count configs; later entries in this file will
say whether the gap narrows, worsens, or stays flat as those knobs change,
which is itself evidence for or against the unbounded-pass hypothesis above
(e.g. if larger `traverse_batch_size`/`modify_batch_size` values change
`relocation_batches`/`candidates_requeued` in a consistent direction, that's
a strong signal either way).

### Config 2 (`b4_t1`, batch=4, exec/client threads=1)

`build: 688.9 ms` (vs. config 1's 1,712,981 ms) -- confirms the `--load-index`
build-once optimization works as intended: a ~2,487x reduction for the
build phase alone, ~4h07m52s -> 1h52m28s for the whole config now that it
skips rebuilding.

| | batch=1 (config 1) | batch=4 (config 2) | change |
|---|---|---|---|
| insert ms/op | 15.7161 | 11.1509 | 29% better |
| stream search ms/op | 13.1904 | 5.8727 | 55% better |
| relocation_batches | 19 | 16 | ~flat |
| candidates_requeued | 1,609,806 | 838,307 | 48% lower |
| candidates_rejected | 12,911 | 12,909 | flat |
| regions_promoted / evicted | 5,690 / 5,422 | 5,217 / 4,848 | slightly lower |
| gpu_bytes_allocated | 73,453,384 | 101,312,640 | fuller use of the 100 MiB budget |

Batch size clearly helps, substantially -- but `relocation_batches` staying
essentially flat (16 vs 19) while `candidates_requeued` roughly halves
suggests the improvement is coming from `OpScheduler` merging more
traverse/modify requests into fewer, larger adapter calls (so fewer distinct
traversal-touch events get recorded and turned into promotion candidates in
the first place), **not** from the Coordinator itself finalizing
proportionally more, smaller relocation passes. That's a meaningful
refinement of the "unbounded per-pass batch size" hypothesis above: the
per-pass size itself doesn't look like it's shrinking as
`traverse_batch_size`/`modify_batch_size` grows (both configs are stuck
around 16-19 total passes for the whole run either way) -- what's shrinking
is the total candidate volume feeding into those passes. Still consistent
with `max_promotion_bytes_per_pass`/`max_eviction_bytes_per_pass` being
unbounded (a smaller candidate volume into an equally-unbounded pass just
means each of those ~16-19 passes has less to chew through, not that the
per-pass cap itself is doing anything) -- but it reframes where the highest-
leverage fix might be: reducing candidate volume at the source may matter as
much as, or more than, capping pass size after the fact. Both remain
untested; more configs across the batch-size sweep (especially 128/256) will
say whether this trend continues or plateaus.

### Config 3 (`b8_t1`, batch=8, exec/client threads=1): the batch-size benefit plateaus

| | batch=1 | batch=4 | batch=8 |
|---|---|---|---|
| insert ms/op | 15.7161 | 11.1509 | 11.6810 |
| stream search ms/op | 13.1904 | 5.8727 | 5.8811 |
| relocation_batches | 19 | 16 | 16 |
| candidates_requeued | 1,609,806 | 838,307 | 875,240 |
| candidates_rejected | 12,911 | 12,909 | 11,249 |

Doubling batch size again (4 -> 8) bought essentially nothing -- insert is
flat to marginally worse, search is flat. Combined with config 2's clear
1 -> 4 improvement, this suggests the benefit from
`traverse_batch_size`/`modify_batch_size` **saturates quickly, around 4**,
rather than scaling with batch size throughout the swept range. The
remaining `batch_size` points (16 through 256) in progress will confirm
whether this is a true plateau or a local dip. Running total, all `batch=1`
config's `elapsed` time included the one-time build; `batch=4`/`batch=8`
(load-index only) both landed at essentially the same ~1h52m wall time
despite batch=8 having no throughput advantage over batch=4 -- i.e. the
*wall-clock* cost of running this config is not improving in lockstep with
"more batching should mean fewer round trips", another data point against
per-pass batch size being the main lever (see config 2's note above).

### Configs 4-5 (`b16_t1`, `b32_t1`): not a clean plateau -- batch=32 regresses

| | batch=4 | batch=8 | batch=16 | batch=32 |
|---|---|---|---|---|
| insert ms/op | 11.1509 | 11.6810 | 11.5506 | 13.8640 |
| stream search ms/op | 5.8727 | 5.8811 | 5.0120 | 6.8642 |
| relocation_batches | 16 | 16 | 15 | 16 |
| candidates_requeued | 838,307 | 875,240 | 860,364 | 1,102,272 |
| candidates_rejected | 12,909 | 11,249 | 12,922 | **35,746** |

batch=16 was the best point so far (search dipped to 5.01 ms/op); batch=32
regresses on every metric, most strikingly `candidates_rejected` nearly
tripling (35,746 vs. ~11-13K at every other batch size tried). Not a clean
monotonic plateau after all -- more data needed (64/128/256, and the
exec_threads=4/8 rows) before concluding whether 16-ish is a real local
optimum or this is run-to-run noise on a single-sample-per-config sweep
(each config here is exactly one run, no repeats -- a real limitation of
this sweep's design worth flagging in the final writeup).

### Config 6 (`b64_t1`, batch=64): batch=32's spike does not recur

insert 12.2655 ms/op, search 5.0416 ms/op, `candidates_rejected=12,473`
(back in the normal ~11-13K range). This confirms the batch=32 spike
(35,746 rejected) was very likely single-run noise, not a trend tied to
batch size -- the metric reverted immediately at the next point tried. All
five points from batch=4 through batch=64 now cluster in the same
~11-14 ms/op insert / ~5-7 ms/op search band, with no further improvement
past batch=4-16 and no further degradation either. Working conclusion so far
for the `exec_threads=1` row: **the batch-size benefit is real going from
1->4 (see config 2), then plateaus** -- batch size beyond ~4-16 does not
meaningfully move insert/search cost in either direction at this scale.

### Configs 1-8 complete: full `exec_threads=1, client_threads=1` row

| batch | insert ms/op | search ms/op | relocation_batches | candidates_requeued | candidates_rejected |
|---|---|---|---|---|---|
| 1   | 15.7161 | 13.1904 | 19 | 1,609,806 | 12,911 |
| 4   | 11.1509 | 5.8727  | 16 | 838,307   | 12,909 |
| 8   | 11.6810 | 5.8811  | 16 | 875,240   | 11,249 |
| 16  | 11.5506 | 5.0120  | 15 | 860,364   | 12,922 |
| 32  | 13.8640 | 6.8642  | 16 | 1,102,272 | 35,746 |
| 64  | 12.2655 | 5.0416  | 16 | 821,243   | 12,473 |
| 128 | 11.3719 | 5.2334  | 18 | 790,575   | 12,017 |
| 256 | 13.2073 | 6.8617  | 17 | 1,326,294 | **334,910** |

Two things stand out once the whole row is in:

1. **The 1->4 jump is the only large, unambiguous win** (batch=1's ~15.7/13.2
   ms/op vs. everything from batch=4 on, clustered around ~11-14 / ~5-7
   ms/op). Batch size above 4 buys nothing further on insert/search cost
   at `exec_threads=1`.
2. **`candidates_rejected` is not flat -- it has two outliers, both at large
   batch sizes**: batch=32 (35,746, ~3x the ~12K baseline) and batch=256
   (334,910, ~28x baseline, and ~10x worse than the batch=32 outlier). With
   batch=64 and batch=128 both landing back in the normal ~12K band *between*
   these two outliers, this no longer looks like simple monotonic scaling
   with batch size, but it also no longer looks like pure noise either --
   two elevated points out of eight, both well above what a single-run
   sampling artifact should produce twice, and the largest batch size tried
   producing by far the largest spike is suggestive of a real effect that
   strengthens at the extremes of the swept range rather than a smooth
   trend. Not enough evidence yet to say more than that -- this sweep design
   (one run per config, no repeats) cannot distinguish "batch size directly
   causes more rejections at high values" from "some other time-varying
   factor (e.g. system load from whatever else was running, thermal
   throttling, memory pressure at ~3.7GB+ process RSS) correlates with when
   the batch=32 and batch=256 runs happened to execute." Worth a repeat run
   of just batch=32 and batch=256 with nothing else running, if this needs
   to be resolved with confidence -- flagged as a possible follow-up rather
   than pursued now, to keep the sweep moving through the remaining
   exec_threads=4/8 rows.

Interesting alignment: `candidates_rejected` correlates with `insert`/
`search` cost getting *worse* at both outlier points (batch=32: 13.86/6.86;
batch=256: 13.21/6.86 -- both among the worst 3 of the 8 rows), while the
best points (batch=16 at 11.55/5.01, batch=128 at 11.37/5.23) have
`candidates_rejected` near the low end. This is consistent with rejected
promotion candidates being expensive to process too (not free just because
they don't end up resident) -- another data point for the "candidate volume,
not just per-pass batch size, drives cost" framing from configs 1-3 above.

## `exec_threads=4, client_threads=4` row begins: thread count matters far more than batch size

### Config 9 (`b1_t4`, batch=1, exec/client threads=4)

| | batch=1, threads=1 (config 1) | batch=1, threads=4 (config 9) | change |
|---|---|---|---|
| insert ms/op | 15.7161 | 6.8181 | **2.3x better** |
| stream search ms/op | 13.1904 | 4.2949 | **3.1x better** |
| relocation_batches | 19 | 10 | fewer |
| candidates_requeued | 1,609,806 | 360,874 | **4.5x lower** |
| candidates_rejected | 12,911 | 13,244 | flat |
| elapsed (whole config) | 14,872s (incl. one-time build) | 4,847s | much faster even accounting for no rebuild |

Going from 1 to 4 worker/client threads, at the *same* batch=1, beats the
*best* batch-size-only improvement seen in the entire `exec_threads=1` row
(batch=16's 11.55/5.01 ms/op) by a wide margin. This is the single largest
improvement found in this sweep so far, larger than any batch-size change
tried. Preliminary read: `exec_threads`/`client_threads` -- i.e. genuine
concurrent request handling -- is a substantially higher-leverage knob than
`traverse_batch_size`/`modify_batch_size` for closing the gap with raw
hnswlib at this scale. The remaining `batch x threads=4` and `threads=8`
points will show whether this holds across the batch range too, and whether
`threads=8` improves further or itself plateaus the way batch size did.

### `exec_threads=4` row so far (configs 9-12): mixed but promising

| batch | insert ms/op | search ms/op | relocation_batches | candidates_requeued | candidates_rejected |
|---|---|---|---|---|---|
| 1  | 6.8181 | 4.2949 | 10 | 360,874 | 13,244 |
| 4  | 7.6225 | 4.0935 | 20 | 444,444 | 43,501  |
| 8  | 6.4728 | 3.5545 | 10 | 256,652 | 17,761  |
| 16 | 8.2331 | **1.4926** | 7 | 284,089 | 14,339 |

Every point in this row beats every point in the `exec_threads=1` row on
both insert and search -- confirms threads=4 is a strictly better regime
than any batch-size tuning at threads=1. Within the row itself, batch=16's
search result (1.4926 ms/op) stands out sharply -- within 2x of raw
hnswlib's own 0.7682 ms/op search cost at this scale, the closest this
entire investigation has gotten arachne's search to raw's. insert doesn't
follow the same pattern (batch=16 is actually the worst insert point in this
row so far) -- insert and search appear to respond differently to batch
size once thread count is raised, unlike the `exec_threads=1` row where they
moved together. `candidates_rejected` continues to be noisy (43,501 at
batch=4 here, echoing the unexplained spikes seen in the threads=1 row) --
same caveat as before: no repeat runs, so a real batch=4-specific effect
can't yet be distinguished from noise.

### Config 14 (`b64_t4`, batch=64, exec/client threads=4): search beats raw hnswlib

`stream search: 0.5381 ms/op` -- **faster than raw_hnswlib's own 0.7682 ms/op
at this scale.** This is the first (and so far only) point in the entire
investigation where arachne's search cost is *not* a regression against raw
at all, but an improvement. `insert` remains elevated at 8.1665 ms/op (in
the same ~7-8 ms/op band as batch=16/32 in this row -- insert has not
followed search's dramatic improvement).

| batch (threads=4) | insert ms/op | search ms/op | vs. raw search (0.7682) |
|---|---|---|---|
| 1  | 6.8181 | 4.2949 | 5.6x worse |
| 4  | 7.6225 | 4.0935 | 5.3x worse |
| 8  | 6.4728 | 3.5545 | 4.6x worse |
| 16 | 8.2331 | 1.4926 | 1.9x worse |
| 32 | 7.9950 | 1.3644 | 1.8x worse |
| 64 | 8.1665 | **0.5381** | **0.70x -- faster than raw** |

Search cost drops sharply and monotonically as batch size increases within
this `threads=4` row (4.29 -> 4.09 -> 3.55 -> 1.49 -> 1.36 -> 0.54), a much
cleaner trend than anything seen in the `threads=1` row. insert shows no
comparable trend (bounces between 6.47 and 8.23 with no clear direction).
This asymmetry is itself informative: whatever is driving search's
improvement with larger batches at threads=4 (larger `traverse_batch_size`
merging more concurrent search requests into fewer, more efficient
`traverseDevice()`/`traverseHost()` adapter calls, most plausibly) is not
the same mechanism helping (or not helping) insert -- consistent with
search and insert going through separate `TraverseTask`/`ModifyTask`
pipelines in `OpScheduler` that share the Coordinator/RegionManager backend
but not the batching path itself. Remaining points (128, 256) will show
whether search keeps improving past this or has already bottomed out near
Region-traversal's own floor cost.

## `exec_threads=4` row complete: full table, and the best result of the entire sweep

| batch | insert ms/op | search ms/op | relocation_batches | candidates_requeued | candidates_rejected |
|---|---|---|---|---|---|
| 1   | 6.8181 | 4.2949 | 10 | 360,874 | 13,244  |
| 4   | 7.6225 | 4.0935 | 20 | 444,444 | 43,501  |
| 8   | 6.4728 | 3.5545 | 10 | 256,652 | 17,761  |
| 16  | 8.2331 | 1.4926 | 7  | 284,089 | 14,339  |
| 32  | 7.9950 | 1.3644 | 10 | 266,256 | 13,041  |
| 64  | 8.1665 | 0.5381 | 10 | 173,456 | 13,635  |
| 128 | 8.1226 | 1.4953 | 6  | 286,886 | 13,997  |
| 256 | **1.5535** | **0.5232** | 9 | **9,181** | **494,391** |

(raw_hnswlib reference: insert 1.8885 ms/op, search 0.7682 ms/op.)

**batch=256/threads=4 beats raw hnswlib on both insert and search** -- the
only config in the entire sweep where that happens for insert, and the best
absolute number for both metrics. The `candidates_requeued`/
`candidates_rejected` pattern flips completely relative to every other
config tried: `requeued` collapses to 9,181 (an order of magnitude below
every other point, which all sit in the 170K-450K range), while `rejected`
explodes to 494,391 (more than an order of magnitude above every other
point's 13-44K range, and larger than every prior "spike" seen so far,
including batch=256/threads=1's own 334,910).

**Working interpretation**: this looks like a genuine phase change, not
noise. A `requeued` candidate is one that got pulled into a
`buildRelocationPlan()` pass, couldn't be resolved that pass, and goes back
to the policy to be offered again later -- i.e. it costs Coordinator time
*more than once*. A `rejected` candidate is decided (by `evaluateAdmission()`)
in a single pass and does not come back. If a large `traverse_batch_size`
means the Coordinator sees a bigger, more complete picture of concurrently-
pending demand each pass, `cost_aware`'s admission logic may be able to make
a confident reject-and-move-on decision far more often, instead of the
uncertain "maybe next pass" requeue that dominates at every other batch
size tried. Fewer repeat attempts per candidate would directly explain both
the collapsed `candidates_requeued` and the dramatic throughput win --
*rejecting decisively is cheaper than retrying indefinitely*, and only the
very largest batch size in this sweep reached whatever threshold makes that
kick in. This reframes the "possible improvement directions" from the 1M
config-1 analysis above: rather than (only) capping per-pass size
(`max_promotion_bytes_per_pass`), the more direct lever suggested by this
result is *enlarging* the batch of demand `cost_aware` gets to evaluate at
once, so it rejects with confidence sooner rather than requeuing
repeatedly. These two ideas are not contradictory (a pass could still be
capped in *bytes moved* while being uncapped in *candidates considered for
admission*), but they point in different tuning directions and this result
was not anticipated by the earlier analysis -- worth reconciling before
recommending a specific config change.

Still unconfirmed by direct profiling (same caveat as before): this is
read off aggregate counters, not a traced/attributed measurement of where
time actually goes inside a `cost_aware` admission decision at batch=256.
Given how large and reproducible-looking this effect is, it is the single
best candidate in this whole investigation for that kind of direct
follow-up profiling.

### Config 21 (`b32_t8`, batch=32, exec/client threads=8): a different kind of anomaly -- real thrashing, absorbed without a throughput collapse

`regions_promoted=88,703 regions_evicted=88,325 anchor_evictions=88,325
relocation_batches=233 candidates_rejected=348,739` -- every one of these is
15-40x larger than this config's neighbors in the sweep (every other config
so far sits in the 2,000-7,000 promoted/evicted range and 6-20
relocation_batches range). This is qualitatively different from the
batch=256 "reject decisively, requeue less" pattern seen twice before
(configs 8 and 16): there, `regions_promoted`/`regions_evicted` stayed
normal while only `candidates_rejected` spiked. Here, the *entire* Region
promotion/eviction machinery ran at 15-40x its usual volume -- this looks
like genuine thrashing (Regions being promoted and evicted repeatedly),
not just more candidates being decided faster.

And yet: `insert=6.0616 ms/op`, `search=1.6774 ms/op` -- both perfectly
respectable numbers, in the same range as several non-anomalous configs in
this row. **The system absorbed a 15-40x spike in internal churn without a
corresponding collapse in measured throughput.** Whether that means the
extra promotion/eviction work is genuinely cheap when it happens (so
thrashing barely matters at this scale/budget), or whether this run got
lucky in some other way not visible in these aggregate counters, is not
resolved here. Combined with the batch=32 spike seen at `threads=1`
(config 5) and the batch=4 spike at `threads=4` (config 10), `batch` values
that are not powers-of-two-aligned with `vectors_per_region` (1024) or that
sit at particular ratios to `client_threads` may be worth a closer look --
three of four `candidates_rejected`/promotion-volume anomalies observed so
far (batch=32 at threads=1, batch=4 at threads=4, batch=32 at threads=8)
involve `batch` values that don't evenly divide typical
per-step traffic (10,000) or `vectors_per_region` (1024) as cleanly as
1/8/16/64/128/256 do -- a speculative pattern, not yet confirmed, but a
concrete, checkable hypothesis for follow-up (e.g. does `batch=1024` behave
like the clean powers of two, or like the anomalous points?).

### Config 22 (`b64_t8`, batch=64, exec/client threads=8): new best of the entire sweep

`insert: 2.0653 ms/op` (raw: 1.8885 -- within 1.1x), `stream search:
0.0702 ms/op` (raw: 0.7682 -- **10.9x faster than raw**). This is the best
result found anywhere in this investigation, on both metrics
simultaneously, by a wide margin over the previous best
(batch=256/threads=4, config 16: insert 1.5535/search 0.5232 -- note config
16's insert was nominally better, but config 22's search is dramatically
better and its insert is close). `candidates_rejected=492,011`,
`candidates_requeued=15,234` -- essentially the same signature as the two
earlier "decisive rejection" outliers (config 8: batch=256/threads=1; config
16: batch=256/threads=4), reinforcing that hypothesis: high `rejected` +
collapsed `requeued` correlates directly with the best throughput in every
case observed so far, across all three thread-count rows. Unlike config 21
(batch=32, same threads=8, immediately prior in this same row), which showed
15-40x *promotion/eviction* volume with only moderately elevated rejects,
config 22's `regions_promoted=4,637`/`regions_evicted=4,256` are entirely
normal -- this is the "reject decisively" mechanism operating cleanly,
without the thrashing seen one config over. Two adjacent batch sizes in the
same row (32 and 64) landing in completely different regimes (thrashing vs.
best-of-sweep) is itself notable and reinforces that this sweep's per-config
single-sample design cannot fully separate a true batch-size effect from
run-to-run variance -- but the *recurrence* of the "reject decisively"
signature at three different (batch, threads) combinations, always
correlated with the best throughput each time, is a much stronger pattern
than a single anecdote.

## Full sweep complete (24/24 configs) -- final ranking and synthesis

Ranking every config by total workload processing time (insert + delete +
search combined, excluding build -- the number that matters for "how long
does replaying this workload actually take"), against raw_hnswlib's own
797,169 ms total for the same 900,000 ops:

| rank | config | total ms | vs. raw |
|---|---|---|---|
| 1 | batch=256, threads=4 | 669,627 | **0.84x (16% faster than raw)** |
| 2 | batch=64, threads=8  | 735,997 | **0.92x (8% faster than raw)** |
| 3 | batch=128, threads=8 | 1,314,393 | 1.65x |
| 4 | batch=32, threads=8  | 2,512,086 | 3.15x |
| 5 | batch=8, threads=8   | 2,758,110 | 3.46x |
| 6 | batch=64, threads=4  | 3,033,450 | 3.81x |
| 7 | batch=16, threads=8  | 3,045,450 | 3.82x |
| 8 | batch=256, threads=8 | 3,073,147 | 3.86x |
| 9 | batch=4, threads=8   | 3,121,177 | 3.92x |
| 10 | batch=32, threads=4 | 3,269,653 | 4.10x |
| 11 | batch=16, threads=4 | 3,406,140 | 4.27x |
| 12 | batch=1, threads=8  | 3,419,482 | 4.29x |
| 13 | batch=8, threads=4  | 3,426,313 | 4.30x |
| 14 | batch=128, threads=4 | 3,434,095 | 4.31x |
| 15 | batch=1, threads=4  | 3,783,721 | 4.75x |
| 16 | batch=4, threads=4  | 3,836,059 | 4.81x |
| 17 | batch=128, threads=1 | 5,373,485 | 6.74x |
| 18 | batch=16, threads=1 | 5,477,129 | 6.87x |
| 19 | batch=4, threads=1  | 5,593,915 | 7.02x |
| 20 | batch=64, threads=1 | 5,595,630 | 7.02x |
| 21 | batch=8, threads=1  | 5,725,686 | 7.18x |
| 22 | batch=256, threads=1 | 6,270,235 | 7.87x |
| 23 | batch=32, threads=1 | 6,618,624 | 8.30x |
| 24 | batch=1, threads=1  | 9,142,441 | **11.47x (worst)** |

### Answer to the investigation's original question

**Yes, the original symptom (Arachne substantially slower than raw hnswlib)
is fully resolved -- but only for specific, non-default configurations, and
the default/naive configuration is actually the worst point in the entire
24-point sweep.** The gap between best and worst config is >13x (0.84x vs.
11.47x raw). Two configurations -- `batch=256/threads=4` and
`batch=64/threads=8` -- make the full Controller/OpScheduler/RegionManager
stack *faster* than raw, unbatched, single-threaded hnswlib on the exact
same 1M-base/30-step/900K-op workload. A third (`batch=128/threads=8`) comes
within 1.65x. Every other configuration remains a regression, ranging from
3.15x to the pathological 11.47x at `batch=1/threads=1` (the config nobody
would deliberately choose, but which happens to be every flag's documented
default).

### Consolidated root-cause picture across the whole sweep

Three largely independent levers were identified, and their *interaction*
matters more than any one alone:

1. **`exec_threads`/`client_threads` (genuine concurrency) is the single
   highest-leverage lever.** Every `threads=4` config beats every
   `threads=1` config; every `threads=8` config, with one exception
   (`batch=256/threads=8` at 3.86x, oddly worse than several `threads=4`
   points), beats or ties the `threads=4` row. Real parallel request
   handling reduces both per-op latency (less serialization through
   OpScheduler) and, per the pattern below, changes what the Coordinator
   sees per pass.
2. **`traverse_batch_size`/`modify_batch_size` alone plateaus quickly at
   `threads=1`** (see configs 1-8) -- batch size cannot substitute for
   thread count. But **combined with high thread count, specific
   (batch, threads) pairs unlock a qualitatively different regime**: high
   `candidates_rejected` + collapsed `candidates_requeued`, seen at
   `batch=256/threads=1` (334,910 rejected, no throughput win),
   `batch=256/threads=4` (494,391 rejected, **best insert of the sweep**),
   `batch=64/threads=8` and `batch=128/threads=8` (~490K rejected each,
   **the two lowest search times of the sweep, both far better than raw**).
   The pattern only pays off in throughput when thread count is also high
   enough -- `threads=1` reaches the same high-reject signature without any
   benefit, suggesting the reject-decisively mechanism reduces *Coordinator*
   work, but realizing that as wall-clock throughput requires enough
   worker/client threads to actually keep the adapter busy while the
   Coordinator is no longer the bottleneck.
3. **Not every large batch size hits this regime, and behavior is not
   smooth.** `batch=32` behaved anomalously at all three thread counts
   (elevated `candidates_rejected` at threads=1 and threads=4; a 15-40x
   promotion/eviction thrashing spike at threads=8) without ever reaching
   the "reject decisively" throughput win. `batch=256/threads=8` -- the
   single largest config tried -- unexpectedly reverts to
   mid-pack performance (3.86x) despite `batch=256/threads=4` and
   `batch=64-128/threads=8` all hitting the best regime. This sweep's
   single-run-per-config design cannot fully separate a genuine
   batch-size-dependent effect from run-to-run noise (system load, GC-like
   pauses in RegionManager's own bookkeeping, etc.) -- repeat runs of the
   handful of anomalous/best points would be needed to know which of these
   effects are reproducible and which were one-off.

### Possible improvement directions (final list, still none implemented)

1. **Ship a better default.** The single most actionable, lowest-risk
   finding: `traverse_batch_size=1, modify_batch_size=1, exec_threads=1`
   (every flag's own documented default) is the *worst* of 24
   configurations tried, an 11.47x regression -- while a config only a few
   flags away is *faster than raw*. Even without understanding the
   underlying mechanism, changing recommended/default settings for
   production-like deployments to something in the `threads=4-8`,
   `batch=64-256` neighborhood would likely fix the vast majority of the
   originally-reported gap immediately.
2. **Investigate why `cost_aware` admission behaves so differently at high
   batch+high thread combinations** -- direct profiling of
   `evaluateAdmission()`/`buildRelocationPlan()` at `batch=64/threads=8`
   (best regime) vs. `batch=32/threads=8` (thrashing) vs.
   `batch=256/threads=8` (reverts to mid-pack) would show whether this is a
   genuine, reproducible property of the admission algorithm interacting
   with batch/thread size, or an artifact of this sweep's single-sample
   design. This is now the highest-value next step: the effect size
   (13x range) is too large to leave unexplained.
3. **The `max_promotion_bytes_per_pass`/`max_eviction_bytes_per_pass`
   unbounded-default hypothesis from config 1's analysis is still
   untested directly** -- worth trying alongside (2) above, since it may
   interact with the batch/thread effects rather than being independent of
   them.
4. **Repeat the anomalous and best-performing configs** (`batch=32` at all
   three thread counts; `batch=256/threads=8`; `batch=64/threads=8`;
   `batch=256/threads=4`) at least 2-3 times each to establish which
   effects are reproducible before spending further engineering effort
   chasing a specific mechanism.
5. **The 5 non-`cost_aware` policies were scoped out of this 1M-scale sweep
   entirely** (see the calibration-phase findings above) -- once the
   `cost_aware` picture above is better understood, it may be worth
   revisiting whether `fifo`/`lru`/`clock` (the cheaper-touch policies)
   reach a similar or better "reject/admit decisively at scale" regime, now
   that this sweep shows genuine promotion-candidate-volume effects are
   central to overall cost, which was exactly the axis those simpler
   policies differ on from `cost_aware`.

This concludes the planned `cost_aware`/1M-scale/24-config sweep. Total
wall-clock time for the whole sweep: config 1 started 2026-08-27 04:29:36,
config 24 finished 2026-08-28 ~16:01 (elapsed=3918s from its 14:58:24
start) -- roughly 35.5 hours of background execution across two days.
