# Removed `shouldYieldPass()` and its supporting dead code

Follow-up to [2026-08-28-latency-tracing.md](2026-08-28-latency-tracing.md),
which implemented `ReplacementPolicy::shouldYieldPass()` as a first attempt
at bounding how long one `buildRelocationPlan_collect()` pass could hold
`CostAwareReplacementPolicy::mutex_`, found it regressed things (defeated
`buildEvictionCandidates()`'s per-pass cache), and reverted its default to
off in favor of the `touch_queue_`/delete-path fixes that solved the actual
root cause instead (removing the client-facing threads' need to ever
contend for that mutex, rather than bounding how long the Coordinator could
hold it). The mechanism was left in place, disabled, "in case a future
workload still benefits from bounding pass duration for some other reason."

This entry removes it outright, at the user's request, now that both real
fixes are verified and the mechanism has sat unused since. Scoped
deliberately narrow: only code that existed *for* `shouldYieldPass()` --
nothing else touched.

## What was removed

- `ReplacementPolicy::shouldYieldPass()` (base class virtual, default
  `false`) and `CostAwareReplacementPolicy::shouldYieldPass()` (its only
  override).
- `CostAwareReplacementConfig::max_pass_duration`.
- `RelocationBatchContext::candidates_examined_this_pass` and `::pass_elapsed`
  -- confirmed (by grep before removing) these two fields had no reader
  anywhere except `shouldYieldPass()` itself; `RelocationBatchContext`'s
  other fields (`batch_sequence`, `selected_promotions`,
  `selected_incremental_bytes`, `available_bytes`, `gpu_budget_bytes`,
  `max_promotion_bytes`) are still live -- `evaluateBatchAdmission()` uses
  the same struct and was left untouched.
- `buildRelocationPlan()`'s (`region_manager.cpp`) collect-loop yield check
  (the `RelocationBatchContext yield_context{...}` construction and the
  `if (replacement_policy_->shouldYieldPass(...))` block immediately before
  each candidate pull), and the now-dead `candidates_examined`/`pass_start`/
  local `Clock` alias that existed only to feed it. The second
  `RelocationBatchContext batch_context{...}` construction (feeding
  `evaluateBatchAdmission()`) stays, with its argument list trimmed to match
  the struct's smaller shape.
- 5 unit tests in `replacement_policy_test.cpp`
  (`ShouldYieldPassIsFalseWhenMaxPassDurationIsZero`,
  `ShouldYieldPassIsFalseBeforeConfiguredDurationElapses`,
  `ShouldYieldPassIsTrueOnceConfiguredDurationElapses`,
  `ShouldYieldPassDefaultConfigNeverYields`,
  `ShouldYieldPassIgnoresEverythingButPassElapsed`) and 1 in
  `region_manager_coordinator_test.cpp`
  (`ShouldYieldPassSplitsOneWaitIdleIntoMultiplePassesWithoutRequeuing`,
  plus its `YieldAfterOneCandidatePolicy` test double).

## What was deliberately left alone

Everything else from the same investigation stays: `touch_queue_`
(`onAnchorTouched()`'s own redesign), `MintAnchorId()`, `isKnownAnchor()`,
`commitRemove()`'s no-longer-calling-`releaseAnchor()`, and the
`buildRelocationPlan_collect`/`buildAdmissionContext()` deep-copy finding
from entry 4 (still open, unrelated to this cleanup -- see that entry).

## Verification

Full suite: **364/365** (371 minus the 6 removed tests; same single
pre-existing, unrelated failure as every entry in this investigation).
Confirmed clean across 4 consecutive runs (no flakiness this time). Grepped
the whole `cpp/` tree afterward for `shouldYieldPass`, `max_pass_duration`,
`pass_elapsed`, and `candidates_examined_this_pass` -- zero remaining hits
outside this investigation's own historical report files (left untouched,
per this README's convention).

Not re-run at 1M scale: this removes an already-disabled-by-default,
unused-in-practice code path -- `shouldYieldPass()`'s default was always
`false`/unlimited, so no config in any prior 1M-scale entry ever exercised
it, and its removal changes no runtime behavior for any existing caller.
