# 10M-scale root cause: the Coordinator's per-pass victim-selection copy,
# not anything cost_aware-specific

Follow-up to the CostAware refinement discussion in this conversation. Before
attempting the originally-requested 6-policy x align/nonalign/random x
24-config x 10M sweep, a calibration pass exposed a severe, unexplained
wall-clock gap that made the sweep's numbers untrustworthy on their own:
`cost_aware` and `fifo` calibration runs at 10M/50-step both took far longer
than their own printed op-time sums accounted for (75min and 102min wall
clock against ~5-6min of measured insert/delete/search time), and the size
of that gap varied wildly run to run (8min to 70min for nominally the same
config) with no correlation to backlog size. This entry root-causes that gap.

## Method

Two new `ARACHNE_ENABLE_TRACING`-only scopes added (tracing-only change, no
logic touched -- see "What changed" below), rebuilt `build-trace/`, reran the
same 10M/50-step/single-thread/cost_aware config that showed the gap, with
per-line wall-clock timestamps on stdout to precisely bound where the gap
falls (loop vs. post-loop) before drilling into the trace CSVs for the
post-loop portion specifically.

## Finding 1: the gap is entirely post-loop, not distributed across steps

Timestamped run: step 1 at t=0, step 50 at t=325s (matches the sum of
insert/delete/search timers almost exactly -- the streaming loop itself is
never the problem). The `=== arachne_controller summary ===` line -- printed
only after `RunArachneController()` returns and its local `Controller`/
`RegionManager` are destroyed -- didn't appear until t=4,408s. **The entire
4,083-second (68-minute) gap is inside `Controller`/`RegionManager`
shutdown**, specifically `RegionManager::shutdown()`'s forced final drain.

## Finding 2 (ruled out): `ARACHNE_LOG_INFO`'s unconditional `fmt::format()`

`ARACHNE_LOG_INFO(...)` expands to `RAFT_LOG_INFO("%s", ::fmt::format(...).c_str())`
-- the `fmt::format()` call is a macro *argument*, evaluated unconditionally
before the logger's own runtime level check ever runs, meaning `--quiet-logs`
(which only raises that runtime level) suppresses the printed *output*, not
this formatting cost. This looked like a strong candidate given the sheer
call volume in `buildRelocationPlan_collect`'s loop (once per candidate
examined, ~1,048,065 times in this run) -- wrapping just that log statement
in its own scope (`buildRelocationPlan_collect_log`) measured **0.72 seconds
total** across all million-plus calls. Genuinely negligible. Hypothesis
rejected by direct measurement, not assumption.

## Finding 3 (the actual cause): `buildRelocationPlan_evict`'s per-pass deep
copy, at 44x the pass count 1M scale ever reached

| scope | count | sum | mean/call |
|---|---|---|---|
| `buildAdmissionContext` | 1,048,065 | 15.7s | 0.015ms |
| `evaluateAdmission_full` (new, whole-function) | 1,048,065 | 630.0s | 0.60ms |
| `evaluateAdmission_locked` (pre-existing, locked-section-only) | 982,017 | 629.8s | 0.64ms |
| `buildRelocationPlan_collect` (the whole collect phase) | 4,072 | 697.1s | 171.2ms |
| **`buildRelocationPlan_evict`** | **4,069** | **2,130.0s** | **523.5ms** |
| `RegionManager::make` + `DeviceRegionPool::*` | — | ~3s | negligible |
| `processRelocationBatch` (collect + evict + execute, the whole pass) | 4,072 | 2,869.3s | 704.6ms |

`collect`(697s) + `evict`(2,130s) + tiny ≈ `processRelocationBatch`'s own
2,869s, within ~1%. Nothing left unaccounted for. `evaluateAdmission_full`
and `_locked` being nearly identical confirms the fast-path (pre-lock) checks
in `evaluateAdmission()` cost essentially nothing -- the entire admission
cost is the already-known, already-cheap-per-call locked eviction scan.

**`buildRelocationPlan_evict` is the dominant cost, and its per-call
duration is remarkably *consistent* (mean 523.5ms, max only 568.3ms across
4,069 calls -- a narrow band, not a few catastrophic outliers)**, unlike
`buildRelocationPlan_collect`'s own highly variable per-call cost (max
28,661ms on some calls, far above its 171ms mean). A flat, consistent
per-call tax times a very large call count is exactly what a genuine O(1)
cost multiplied by (many more calls than expected) looks like -- and that's
what this is:

```cpp
// buildRelocationPlan()'s victim-selection phase, region_manager.cpp
std::vector<EvictionCandidate> candidates = *eviction_candidates_cache;
candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
    [&promoted_anchors](const EvictionCandidate& candidate) {
      return promoted_anchors.contains(candidate.anchor_id);
    }), candidates.end());
```

This is the **same architectural pattern** as
[2026-08-30-eviction-candidates-shared-ptr.md](2026-08-30-eviction-candidates-shared-ptr.md)'s
`AdmissionContext::eviction_candidates` finding -- a cached
`std::vector<EvictionCandidate>` (each element itself owning a
heap-allocated `group_members` vector) being deep-copied out of a
pass-lifetime cache -- except that entry's own text explicitly reasoned this
particular copy site was safe to leave alone: *"called once per pass, not
once per candidate, so it was never the cost the shared_ptr change above
targets."* That reasoning assumed pass count stays roughly what it was at
1M scale (16-97 across every config measured there). It does not hold at
10M scale: **this run alone made 4,069 separate passes that reached the
evict phase** -- 44-250x more than any 1M-scale measurement in this whole
investigation. A per-pass cost that was worth ~46 seconds total at 92 passes
(1M scale, never separately noticed against a ~2.5-minute whole-run budget)
is worth 35.5 minutes at 4,069 passes -- the copy itself didn't get more
expensive, the number of times it's paid did.

## Open sub-question: why does pass count itself explode at 10M scale?

Not answered by this entry. `processRelocationBatch` count went from
16-97 (every 1M-scale config measured this whole investigation) to 4,072
here -- a mechanism-level explanation for *that* (rather than just "it's
bigger, so naturally more happens") is still open. Plausible contributing
factors, none confirmed: `eviction_candidates_cache`'s own larger size at
this budget (~1,954 residency-capacity regions vs. ~800 at 1M, from budget
size alone) meaning fewer candidates get admitted per pass before the
budget's own byte accounting forces a new one; the 100ms `trigger_interval`
coalescing window coalescing a smaller fraction of the arrival stream at
this workload's particular candidate-arrival rate; or something in how
`--limit-steps 50` here vs. `--limit-steps 30` at every 1M-scale run
changes the candidate-arrival cadence. Worth its own investigation before
(or instead of) assuming a fix to the copy alone resolves the pass-count
side too.

## Not cost_aware-specific

Nothing in `buildRelocationPlan_evict`'s cost depends on which
`ReplacementPolicy` is active -- `RegionManager` builds and copies
`eviction_candidates_cache` identically regardless, and every built-in
policy's own `selectEvictionCandidate()` receives that same `candidates`
vector. The user's own working hypothesis going into this entry (isolate
cost_aware first, since it's "the one with problems") doesn't match the
evidence gathered before this entry either: the `fifo` calibration run
(102 minutes) was *slower* than `cost_aware`'s first calibration run
(75 minutes), despite `fifo` having zero admission-scan cost at all --
consistent with this being a `RegionManager`-side cost shared by every
policy, not something cost_aware's own density/hysteresis logic causes.
The originally-planned 6-policy trace sweep was paused once this single
root cause looked general enough to explain the pattern already seen
across both policies tried -- not run to completion, since repeating it
6 more times would very likely just reconfirm the same finding at high
time cost.

## What changed (tracing-only, per this entry's own scope)

- `RegionManager::buildAdmissionContext()`: wrapped in a new
  `ARACHNE_TRACE_SCOPE("RegionManager", "buildAdmissionContext")` (was
  previously untraced as its own scope).
- `CostAwareReplacementPolicy::evaluateAdmission()`: wrapped in a new
  `ARACHNE_TRACE_SCOPE("CostAwareReplacementPolicy", "evaluateAdmission_full")`
  covering the whole function (the pre-existing `evaluateAdmission_locked`
  scope only ever covered the locked tail, deliberately, per its own
  comment -- both are kept, for the fast-path-vs-locked-section comparison
  this entry needed).
- `RegionManager::buildRelocationPlan()`'s per-candidate log statement:
  wrapped in a new `ARACHNE_TRACE_SCOPE("RegionManager", "buildRelocationPlan_collect_log")`,
  isolating `ARACHNE_LOG_INFO`'s own cost from the rest of the loop body.
- No admission/eviction decision logic, no core algorithm, touched at all.
  Verified: both `build/` and `build-trace/` compile clean; full unit suite
  365/366 (same single pre-existing, unrelated failure as every entry in
  this investigation) -- unaffected, as expected for trace-scope-only
  additions that are no-ops outside `ARACHNE_ENABLE_TRACING` builds.

## Suggested fix direction (not implemented in this entry)

Same shape as the admission-side fix: avoid the full-vector deep copy.
Since this site *mutates* its copy (removing already-promoted-this-pass
anchors) unlike the admission site's read-only use, a straight shared_ptr
swap isn't a drop-in fix here -- but `promoted_anchors` is typically small
(bounded by how many candidates one pass admits), so filtering it out
lazily during `selectEvictionCandidate()`'s own scan (passing the shared,
uncopied cache plus the small exclusion set, rather than materializing a
filtered copy up front) would avoid paying for the full ~1,954-entry copy
(each with its own heap-allocated `group_members`) on every single pass.
Not attempted here -- this entry's scope was root-causing the gap, not
fixing it.

## Trace locations (not copied into git, same reasoning as every prior entry)

`/tmp/arachne_trace_10m_cost_aware_v2/` (the traced run this entry's numbers
come from). The untraced calibration runs
(`/tmp/arachne_10m_sweep/calibration/*.log`,
`/tmp/arachne_10m_sweep/trace_cost_aware_single*.log`) are plain stdout
captures, small, kept alongside for the timestamp analysis this entry cites.
