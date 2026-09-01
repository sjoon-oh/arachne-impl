# Fixing entry 11's root cause: excluded_anchors instead of a filtered copy

Follow-up to [2026-08-31-10m-scale-coordinator-throughput.md](2026-08-31-10m-scale-coordinator-throughput.md)
(entry 11), which root-caused a 35.5-minute-per-run cost at 10M scale to
`buildRelocationPlan()`'s victim-selection phase deep-copying
`eviction_candidates_cache` (a `std::vector<EvictionCandidate>`, each entry
owning a heap-allocated `group_members`) once per pass, then erase-removing
already-decided anchors out of that copy -- cheap at 1M scale (16-97 passes
ever measured), the dominant Coordinator cost at 10M scale (4,069 passes in
that entry's traced run). That entry stopped at root-causing and explicitly
did not implement a fix. This entry implements and verifies it.

## The fix

Replaced the per-pass filtered copy with a small, growable exclusion set
threaded through `ReplacementPolicy::selectEvictionCandidate()`'s existing
interface, alongside the single `excluded` id every policy already checked:

```cpp
virtual std::optional<VectorId> selectEvictionCandidate(
    VectorId excluded, std::size_t required_bytes, const std::vector<EvictionCandidate>& candidates,
    const std::unordered_set<VectorId>& excluded_anchors = {}) = 0;
```

`RegionManager::buildRelocationPlan()`'s victim-selection loop now shares
`*eviction_candidates_cache` directly (no copy) with every
`selectEvictionCandidate()` call, and grows a local `excluded_anchors` set
instead of erasing out of a copy -- first seeded with anchors already
promoted this pass, then with each victim group's members as the while loop
selects them (only on the path that actually keeps the group -- the
`max_eviction_bytes_per_pass` rollback branch, which un-commits a group and
`break`s, preserves its original ordering: the group is never added to
`excluded_anchors` on that path either, exactly matching the old code's
behavior of never erasing it from the working copy on that path). All 6
built-in policies (`Fifo`, `Lru`, `Lfu`, `Clock`, `TwoQ`, `CostAware`) were
updated to check `!excluded_anchors.contains(id)` alongside their existing
`id != excluded` check, in the same scan -- the selected victim is unchanged
(same predicate, evaluated during the scan instead of pre-filtered into a
copy first); only how "already decided" is represented changed.
`excluded_anchors` defaults to `{}` so any caller with nothing beyond the
single `excluded` id (every existing test, anything not routing through this
specific loop) is unaffected.

## Correctness

- **9 new unit tests** (`replacement_policy_test.cpp`), one pair (or one,
  for CostAware) per policy, each mirroring that policy's existing
  `excluded`-skip test but driving the skip through `excluded_anchors`
  instead: `Fifo` (skip 2 via the set, nullopt when the set covers
  everything), `Lru` (same pair), `Lfu` (skip across frequency buckets via
  the set), `Clock` (same pair, still respecting the reference-bit
  second-chance semantics), `TwoQ` (skip across `a1in_`/`am_` via the set),
  `CostAware` (a tie-break case: two identically-scored victims, excluding
  the first-seen forces the second, proving the set is actually consulted
  and not just coincidentally matching tie-break order).
- **Full suite: 374/375 passing** (9 more tests than before this entry;
  same single pre-existing, unrelated failure --
  `HnswlibIndexInsertAfterPromotionTest.DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot`
  -- as every entry in this investigation).
- **Existing multi-victim-per-pass integration coverage, unmodified,
  still passing**: `RegionManagerCoordinatorTest.StrictBatchEvictsEnoughVictimsBeforePromotingWholeFootprint`
  and `.MoreThanHalfGpuMemoryIsReplacedAsOneLargeNearFitBatch` both drive
  `buildRelocationPlan()`'s while loop through 2+ iterations in a single
  pass -- exactly the path where `excluded_anchors` has to accumulate
  correctly across iterations (seeded from `promoted_anchors`, then grown by
  each selected group) for eviction to still free enough capacity. Both
  passed unchanged, which is strong evidence the loop-accumulation logic is
  right, not just the single-call contract each new unit test checks in
  isolation. `GroupEvictionReclaimsRegionSharedByMultipleAnchorsWhenCapAllowsIt`
  and `AdmissionContextSharesTheSameEvictionCandidatesSnapshotAcrossOnePass`
  (group semantics and the admission-side shared cache, both untouched by
  this fix) also unaffected.
- **`build/` and `build-trace/` both compile clean.**
- **10M re-run's recall matches entry 11's own number exactly**: mean
  recall@k = 0.8619 in both, and this run's own final checkpoint (0.8627)
  and mid-run checkpoints track the pre-fix run's within noise -- the fix
  changes *when* the copy happens, never *which* victim gets selected.

## Performance: re-ran the exact config entry 11 traced

Same 10M-scale index (`arachne_10m_index_m16_efc200_cap10500000.bin`), same
`align_workload_a_10m/set_1` default workload, same `--limit-steps 50
--gpu-budget-bytes 256000000 --exec-threads 1 --client-threads 1
--replacement-policy cost_aware --trigger-interval-ms 100 --quiet-logs`,
`build-trace/` binary, fresh `ARACHNE_TRACE_DIR`, same timestamped-stdout
technique as entry 11's Method section.

### Wall clock

| segment | entry 11 (before) | this entry (after) | change |
|---|---|---|---|
| streaming loop (step 1 -> step 50) | 325s | 297s | -- (noise, not the target) |
| **post-loop gap (step 50 -> summary line)** | **4,083s (68.0 min)** | **7s** | **~583x** |
| **total (step 1 -> summary line)** | **4,408s (73.5 min)** | **304s (5.1 min)** | **~14.5x** |

The post-loop gap entry 11 root-caused is, for practical purposes, gone --
7 seconds is in the same range as the `RegionManager::make()`/
`DeviceRegionPool::*` "tiny" bucket entry 11's own table already treated as
noise.

### Trace scopes: before (entry 11) vs after (this entry)

| scope | count (before -> after) | sum (before -> after) | mean/call (before -> after) |
|---|---|---|---|
| `buildAdmissionContext` | 1,048,065 -> 1,018,002 | 15.7s -> 15.3s | 0.015ms -> 0.015ms |
| `evaluateAdmission_full` | 1,048,065 -> 1,018,002 | 630.0s -> **31.9s** | 0.60ms -> **0.031ms** |
| `evaluateAdmission_locked` | 982,017 -> 962,991 | 629.8s -> 31.8s | 0.64ms -> 0.033ms |
| `buildRelocationPlan_collect` | 4,072 -> **316** | 697.1s -> **52.2s** | 171.2ms -> 165.3ms |
| **`buildRelocationPlan_evict`** | **4,069 -> 35** | **2,130.0s -> 2.8s** | **523.5ms -> 81.1ms** |
| `processRelocationBatch` | 4,072 -> 316 | 2,869.3s -> 75.1s | 704.6ms -> 237.7ms |

`buildRelocationPlan_evict` itself dropped **~750x** in total cost (both
fewer calls -- 116x -- and a 6.4x cheaper mean call, the latter presumably
`ContainsCandidate()`'s linear scan now short-circuiting faster once
`excluded_anchors.contains()` -- an O(1) hash check -- rules a candidate out
before it's ever reached).

### An unplanned second effect: far fewer passes overall, and cheaper admission too

Two numbers in that table are *not* directly explained by "the copy is
gone": `processRelocationBatch`'s own call count dropped 4,072 -> 316 (a
12.9x reduction), and `evaluateAdmission`'s per-call mean dropped ~19x even
though its own code was untouched by this entry (it was already fixed for
its *own* copy in entry 9).

The working explanation, not independently re-verified beyond what these
traces already show: in the pre-fix run, each pass was so slow (523.5ms
just for the evict-phase copy, before any actual eviction work) that
candidates requeued during one pass had more time to pile up before the
next pass ran, which likely both forced more retry passes *and* kept
`eviction_candidates_cache` larger on average across the run (evictions
chronically lagging behind how many Anchors were actually resident) --
`evaluateAdmission_locked`'s cost is a scan over exactly that cache, so a
chronically-larger cache directly taxes every admission call, not just the
evict phase. With the copy gone, passes complete fast enough that this
backlog effect never builds up: fewer, cheaper passes, and a smaller
eviction-candidate list for admission to scan on average. This reads as a
compounding win rather than two unrelated coincidences, but is offered as
the most likely explanation, not a separately-proven one.

This also **substantially narrows entry 11's open question** ("why does
pass count explode 44-250x at 10M scale?") without fully closing it: most
of that explosion (4,072 passes) looks like it was downstream of the slow
copy itself creating a backlog-retry feedback loop, not an independent
scaling effect. 316 passes at 10M is still well above the 16-97 range ever
seen at 1M scale, so *some* real scale-dependent growth remains
unexplained -- just an order of magnitude smaller a mystery than before.

## What changed

- `cpp/include/core/replacement_policy.hpp`: `ReplacementPolicy::
  selectEvictionCandidate()`'s pure-virtual signature gained a 4th
  parameter, `excluded_anchors` (defaulted to `{}`); doc comment rewritten
  to specify the exclusion contract. All 6 concrete policy declarations
  updated to match (each keeping its own `= {}` default so direct-instance
  test call sites without it still compile).
- `cpp/src/core/replacement_policy.cpp`: all 6 `selectEvictionCandidate()`
  definitions updated to check `excluded_anchors` alongside `excluded`.
- `cpp/src/core/region_manager.cpp`: `buildRelocationPlan()`'s
  victim-selection setup and while loop rewritten per "The fix" above --
  no other function touched.
- `cpp/test/unittest/{region_manager_test,region_manager_coordinator_test,
  controller_test}.cpp`: 4 mock `ReplacementPolicy` overrides updated to
  the new 4-parameter signature (all ignore the new parameter except
  `controller_test.cpp`'s delegating mock, which forwards it).
- `cpp/test/unittest/replacement_policy_test.cpp`: 9 new tests, see
  Correctness above.
- This is a real logic change (unlike every trace-only entry before it in
  this investigation) -- scoped narrowly to exactly the one code path entry
  11 identified, with unit + integration test coverage added specifically
  for the new behavior before trusting the 10M re-run's numbers.

## Still open

- The residual ~1-order-of-magnitude pass-count growth (16-97 at 1M -> 316
  at 10M) noted above -- smaller than entry 11's original mystery, but not
  investigated further here.
- Entry 3's flagged `onAnchorEvicted()` mutex-sharing tail and entry 4's
  `buildRelocationPlan_collect` backlog-drain finding remain as previously
  left (unaffected by this entry).
- The originally-requested full 24-config x 3-workload x 6-policy 10M sweep
  is still not run -- this entry's single re-run is a targeted verification
  of the fix, not that sweep.

## Trace locations (not copied into git, same reasoning as every prior entry)

`/tmp/arachne_trace_10m_fix_v1/` (this entry's traced run;
`timestamped_run.log` has the per-line wall-clock timestamps the wall-clock
table above comes from, the CSVs the trace-scope table comes from).
