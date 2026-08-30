# Collapsed the two-method eviction-selection interface into one

Follow-up to the discussion in this same conversation that found
`ReplacementPolicy::selectNextEvictionCandidate()` (a pure-virtual every
policy had to implement) and `selectEvictionCandidate()` (a virtual with a
default that just delegated to it) were, on closer reading, *not* what they
first looked like: every built-in policy already overrode
`selectEvictionCandidate()` with a real, independent implementation (the
same walk as `selectNextEvictionCandidate()`, plus a
`ContainsCandidate(candidates, ...)` filter against the byte/group-aware
list `RegionManager` actually passes) -- meaning the base class's delegating
default was never exercised by any of the 6 built-in policies, and
`selectNextEvictionCandidate()` itself was never called directly by any
production code path (confirmed by grep before touching anything:
`region_manager.cpp` only ever calls `selectEvictionCandidate()`). The
"legacy" method was pure dead weight -- six near-duplicate implementations
nobody used, not a thin compatibility shim.

## What changed

- `ReplacementPolicy`: removed `selectNextEvictionCandidate()` entirely;
  `selectEvictionCandidate(excluded, required_bytes, candidates)` is now the
  sole pure-virtual eviction-selection entry point (was virtual-with-default).
- All 6 concrete policies (`Fifo`/`Lru`/`Lfu`/`Clock`/`TwoQ`/`CostAware`):
  deleted their now-orphaned `selectNextEvictionCandidate()` implementations.
  `selectEvictionCandidate()` itself -- the one actually exercised by
  production code -- was left byte-for-byte untouched in all 6.
  `CostAwareReplacementPolicy`'s deleted method (a simple heat-only global
  scan of `resident_`, structurally different from its group/density-aware
  `selectEvictionCandidate()`, not just missing a filter) was the one
  genuinely distinct implementation lost -- confirmed via grep that nothing,
  including its own test suite, ever called it.
- `region_manager.hpp`: fixed a stale doc comment describing an
  "OutOfCapacity retry loop" that called `selectNextEvictionCandidate()` and
  retried `make()` in place -- this doesn't match current code at all;
  `processRelocationBatch()` actually just marks `OutOfCapacity`/`Deferred`
  results for retry and requeues them for a later pass (fresh
  `buildRelocationPlan()` collect+evict cycle), never retries eviction
  in-place. Also flagged (but did not attempt to fully rewrite) the rest of
  this file's top-of-file flow diagram, which predates
  `buildRelocationPlan()`/`processRelocationBatch()`'s current shape
  (references an old `processPromotions()` name, no collect/evict split
  shown) -- a larger, separate documentation debt.

## Unit tests

46 call sites across 4 test files needed updating for the new
three-argument signature (`excluded, required_bytes, candidates`) --
`replacement_policy_test.cpp` (44, one per policy's own eviction-order
tests -- FIFO/LRU/LFU/Clock/2Q), `controller_test.cpp`,
`region_manager_coordinator_test.cpp`, `region_manager_test.cpp` (test
doubles overriding the interface). A new `Candidates({ids...})` helper
builds the `candidates` argument each renamed test needs -- these tests
exercise each policy's own tracked eviction *order*, not
`ContainsCandidate()`'s filter, so each call site passes every anchor id
that test enqueues (a safe superset the filter never needs to narrow for
what these tests check), not a byte-accurate `EvictionCandidate` model.

Full suite: 365/366 (same single pre-existing, unrelated failure as every
entry in this investigation). All 83 `*ReplacementPolicy*`/`*Fifo*`/`*Lru*`/
`*Lfu*`/`*Clock*`/`*TwoQ*`-filtered tests pass, confirming the 5
non-`cost_aware` policies are unaffected -- expected, since none of their
`selectEvictionCandidate()` bodies were touched.

## 1M-scale check, and an unresolved anomaly

Re-ran the same multi-thread 1M config used throughout this investigation
as a sanity check (this change touches no admission/eviction *decision*
logic, only which function name holds it, so no behavior change was
expected). Recall matched raw within noise in both of two back-to-back
runs, and ms/op figures were in line with or better than every prior round
-- no correctness regression.

**However**: `candidates_rejected` came out consistently around ~31,000 in
both runs, versus a consistent ~497,000 across three independent prior
rounds (this investigation's `2026-08-29`/`2026-08-30` entries) using the
identical CLI flags. Static re-inspection confirmed `evaluateAdmission()`
and `selectEvictionCandidate()`'s bodies are unchanged from the immediately
preceding (shared_ptr) round -- nothing in this entry's diff touches
admission/eviction decision logic at all. Current best explanation: the
Coordinator's relocation passes are triggered on a wall-clock
`trigger_interval`, and this harness's own run-to-run timing variance was
already documented as severe in
[2026-08-28-latency-tracing.md](2026-08-28-latency-tracing.md) ("up to
5.8x different numbers" for an otherwise-identical config) -- a faster
run leaves less real elapsed time for periodic re-triggers, which would
mechanically reduce the total number of admission evaluations a fixed
amount of logical work (900,000 ops) accumulates before the process exits,
independent of any change in admission *logic*. Not confirmed with a
controlled, same-conditions A/B rerun of the pre-this-entry code --
flagged as an open question for whoever picks up the CostAware refinement
work next (see the config-CLI-flags entry immediately following this one),
since it directly affects how to interpret any future before/after
sweep numbers.

## Conclusion

The interface reduces from two eviction-selection entry points to one,
removing genuinely dead code (not a behavior-preserving rename, as
initially assumed -- a correction made mid-investigation once the actual
`.cpp` bodies were read carefully) across the base interface, all 6
policies, and one stale doc comment. Zero logic change to any policy's
actual admission/eviction decisions, confirmed both statically (diff) and
via the unit suite. The 1M-scale admission-count anomaly is noted honestly
as unresolved rather than glossed over, since it's exactly the kind of
signal that could otherwise be mistaken for a real effect in a future
CostAware-tuning sweep.
