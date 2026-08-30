# `AdmissionContext::eviction_candidates`: replaced the per-candidate deep
# copy with a shared_ptr, verified against all 6 policies and at 1M scale

Follow-up to [2026-08-29-anchor-id-independence.md](2026-08-29-anchor-id-independence.md),
which found (but didn't fix) `buildRelocationPlan_collect`'s final-drain
pass becoming the dominant remaining cost once the earlier mutex-contention
fixes landed, and root-caused it to `RegionManager::buildAdmissionContext()`
deep-copying the whole cached `std::vector<EvictionCandidate>` snapshot into
`AdmissionContext::eviction_candidates` once per *candidate* examined
(hundreds of thousands of times in one pass), rather than once per *pass*
the way `buildEvictionCandidates()`'s own cache is. This entry implements
the fix discussed and re-verified (with the user) before applying it.

## Safety re-check before implementing (the point of asking first)

The user asked to re-verify a plain reference would actually be safe before
touching anything. It would not have been: `PlannedPromotion::admission`
(the `AdmissionContext` for an *admitted* candidate) is stored in
`RelocationPlan::promotions` and read again later by
`processRelocationBatch()` -- a separate call, after `buildRelocationPlan()`
(and its function-local `eviction_candidates_cache`) has already returned.
A raw reference into that local cache would dangle at exactly that point.
`std::shared_ptr<const std::vector<EvictionCandidate>>` was used instead --
copying it is an atomic refcount bump (not a deep copy), and its pointee
stays alive for as long as any copy of it does, regardless of whether the
pass-local cache that first built it has gone out of scope.

## What changed

- `AdmissionContext::eviction_candidates`: `std::vector<EvictionCandidate>`
  (owned, deep-copied in) -> `std::shared_ptr<const std::vector<EvictionCandidate>>`
  (shared, refcounted in). Null instead of empty is now the "nothing needed
  evicting" state.
- `RegionManager::buildAdmissionContext()`: builds the cache as a
  `shared_ptr` (once per pass, unchanged), and now *shares* it into each
  candidate's `AdmissionContext` (`context.eviction_candidates = eviction_candidates_cache;`)
  instead of dereferencing and copying it.
- `buildRelocationPlan()`'s local `eviction_candidates_cache` and the
  victim-selection loop further down: same type change; the
  victim-selection site still makes its own *owned* `std::vector` copy
  (`std::vector<EvictionCandidate> candidates = *eviction_candidates_cache;`)
  since it mutates that copy (`erase`/`remove_if`) -- that call site runs
  once per pass, not once per candidate, so it was never the cost this
  entry targets, and is left doing what it already did.
- `CostAwareReplacementPolicy::evaluateAdmission()`'s scan and log line:
  updated for the new null-checked pointer-like access
  (`if (context.eviction_candidates) { for (... : *context.eviction_candidates) ... }`).
  The other 5 built-in policies (fifo/lru/lfu/clock/2q) never override
  `evaluateAdmission()`/`evaluateBatchAdmission()` -- confirmed by grep
  before touching anything -- so this field's type change doesn't reach
  them at all.

## Unit tests

Full suite: **365/366** (same single pre-existing, unrelated failure as
every entry in this investigation). One new test added
(`RegionManagerCoordinatorTest.AdmissionContextSharesTheSameEvictionCandidatesSnapshotAcrossOnePass`),
plus the existing `evaluateAdmission()` mutex-contention test updated for
the new field type. All 82 tests across every `*ReplacementPolicy*`/
`*Fifo*`/`*Lru*`/`*Lfu*`/`*Clock*`/`*TwoQ*` filter pass, confirming the
other 5 policies are unaffected, as expected.

The new test proves *sharing*, not just correctness of values: a custom
always-Reject policy records the raw pointer behind
`AdmissionContext::eviction_candidates` for every candidate examined within
one pass, then asserts all of them are the exact same address. Two real
bugs surfaced and were fixed while writing it, both in the test itself, not
in the production change:

1. **Wrong assumption about `AllocationPolicy::Pooled`'s budget rounding.**
   First attempt configured a data budget smaller than one allocation unit,
   expecting `available < incremental_bytes` to hold from that alone --
   but `DeviceContext::budgetBytes()` under `Pooled` returns
   `arena.totalUnits() * arena.unitBytes()`, i.e. the *configured* budget
   is itself rounded up to a whole unit, silently becoming equal to
   `incremental_bytes` instead of less. Fixed by sizing the Region itself
   at 2 allocation units against a 1-unit budget instead, so the needed
   inequality holds regardless of budget-rounding.
2. **A self-inflicted deadlock**, unrelated to the fix under test: the
   test held its own policy's `mutex_` (via `std::lock_guard`) across the
   `manager.shutdown()` call at the end, while the Coordinator thread being
   joined needed that same `mutex_` for its own exit-path bookkeeping --
   classic lock-held-across-a-join deadlock. Fixed by scoping the lock to
   just the snapshot read, releasing it before `shutdown()`.
3. Also hit, mid-investigation: killing the first hung run with `pkill -9`
   left a *second* mistaken config (`unit_bytes=64` against the *default*
   64 MiB metadata pool, i.e. ~262,144 metadata units) that made
   `DeviceContext` construction itself hang for minutes -- fixed by passing
   an explicitly small `metadata_pool_bytes` alongside the small
   `unit_bytes`, rather than leaving the metadata arena at its default size.

## 1M-scale re-run against raw hnswlib

Same two configs, same flags, same pre-built 1M index as every prior entry.
"Before" below is the immediately preceding round
([2026-08-29-anchor-wraparound-guard.md](2026-08-29-anchor-wraparound-guard.md)'s
own measurement), not re-derived -- same binary flags, same trace scopes.

### The targeted fix: `buildRelocationPlan_collect`'s worst pass

| | single (batch=1, exec=1, client=1) | multi (batch=32, exec=4, client=4) |
|---|---|---|
| max pass duration, before | 602,861.08 ms (10.0 min) | 473,455.09 ms (7.9 min) |
| max pass duration, after | **8,856.91 ms** (8.9 s) | **61,407.19 ms** (61.4 s) |
| reduction | **68.1x** | **7.7x** |

Both dropped dramatically, as expected -- this is exactly the O(candidates
examined x cache size) copy cost the fix removes. `RegionManager-releaseAnchor.csv`
is still absent from both runs' trace output (the entry-4/5 delete-path fix
still holds -- unrelated to, and unaffected by, this change), and
`CostAwareReplacementPolicy-lockwait`'s max stayed at the same negligible
0.0179 ms (single) / 0.0288 ms (multi) as every round since the
`touch_queue_` fix -- nothing client-facing was ever waiting on this cost
in the first place, so this fix's benefit is entirely about how long the
Coordinator's own background work takes to settle, not about unblocking
anything that was stalling client requests.

### A side effect worth reporting honestly: single-thread's relocation
cadence changed shape, and stream-search wall-clock got worse there

| | single (before -> after) | multi (before -> after) |
|---|---|---|
| `relocation_batches` (Coordinator passes) | 16 -> **97** | 8 -> 8 (unchanged) |
| insert ms/op | 0.4381 -> 0.4200 | 0.1786 -> 0.1847 |
| search ms/op | 0.0671 -> **0.1269** (1.9x worse) | 0.0176 -> 0.0252 (1.4x worse, still far below raw's 0.1218) |
| delete ms/op | 0.0004 -> 0.0004 | 0.0005 -> 0.0005 |

`regions_promoted`/`regions_evicted`/`anchor_evictions` are identical
before and after in both configs (single: 3810/3429/3429; multi:
1524/1143/1143) -- the *total* amount of real promotion/eviction work is
unchanged, only how it's scheduled over time is different. In the
single-thread config specifically, removing the copy cost let the
Coordinator's cadence collapse back toward its natural
`trigger_interval`-driven rate (many small passes) instead of being
effectively throttled into a few enormous ones by how long each pass used
to take -- `RegionManager-make`/`DeviceRegionPool-tryAllocate`/`-acquire`
call counts are all within noise of their prior values (confirming the
same total GPU work, not more of it), but it's now spread across 6x more
Coordinator wake-ups. `OpScheduler-executeTraverseBatch`'s own mean rose
from 0.0724 ms to 0.1045 ms across all 630,000 calls (not just a tail
outlier) -- consistent with `--exec-threads 1`'s single execution worker
now sharing time with a Coordinator that interrupts it more often, in
smaller pieces, rather than rarely in one long stretch. This is a
hypothesis, not something separately instrumented to confirm this session --
flagged as such.

The multi-thread config's `relocation_batches` stayed flat (8 -> 8, exactly
the same cadence as before), and its search regression is far milder (1.4x,
and still 4.8x *faster* than raw hnswlib) -- consistent with this being
specific to `--exec-threads 1` sharing one worker between traversal
execution and Coordinator GPU work, not a general property of the fix.
**Net assessment**: the realistic/recommended multi-thread configuration
shows a clean win with no concerning regression; the single-thread
configuration trades a large, real background-cost win for a real, smaller,
isolated foreground cost -- worth knowing about, not obviously worth
reverting the fix over (recall and every other measured quantity stayed
correct and consistent in both configs).

## Conclusion

The fix does exactly what it was designed to do -- eliminates the
O(candidates x cache size) copy cost, confirmed by both configs' worst-pass
duration (68x and 7.7x reductions) and by the new regression test proving
the sharing directly, not just the resulting numbers. Confirmed harmless to
the 5 non-`cost_aware` policies (82/82 policy tests passing, none of them
touch the changed field). One genuine, non-obvious side effect found and
reported plainly rather than glossed over: in the single-thread config
only, a faster Coordinator now interleaves its (unchanged-in-total) GPU
work more finely with traversal execution on the one shared worker thread,
measurably raising stream-search wall-clock time there (not in the
multi-thread config, and not via any correctness or recall regression in
either). `StressIndexStage2Test.EvictionCyclingPreservesCorrectnessUnderTinyGpuBudget`
recurred at roughly its previously-characterized rate (2 failures across 7
full-suite runs this session, always passing standalone) -- consistent with
the CUDA cold-start artifact already investigated and dismissed in
[2026-08-29-anchor-wraparound-guard.md](2026-08-29-anchor-wraparound-guard.md),
not re-investigated further here.
