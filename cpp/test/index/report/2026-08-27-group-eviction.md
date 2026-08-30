# Group-based eviction: design, implementation, and benchmark result

Status: implementation complete and merged; benchmark validation complete
with a nuanced result (see "Outcome" below). Retroactive writeup -- this
work was done earlier in the same investigation, before this `report/`
folder existed.

## Symptom that started this

Comparing `raw_hnswlib` (unmodified hnswlib, no Arachne machinery) against
`arachne_controller` (the full Controller/OpScheduler/RegionManager stack)
on a real SIFT1B streaming workload, under a GPU budget tight enough to
force real eviction churn, Arachne's insert/search throughput was
substantially worse than raw hnswlib's -- not just "some overhead", but
close to or worse than raw depending on batching config.

## Root cause found

Direct log analysis of `CostAwareReplacementPolicy::evaluateAdmission()`
rejections (with `--quiet-logs` off, at 100K-scale) showed **97.4%** of all
promotion rejections were `"no eligible victim found among N eviction
candidate(s)"` -- i.e. Arachne wasn't being outcompeted on cost/heat
scoring, it was refusing to even try to evict, most of the time.

Traced to `RegionManager::buildEvictionCandidates()`'s original rule:
a Region only counted as reclaimable if it had exactly one dependent Anchor
(`dependents_it->second.size() == 1`). But `RegionManager::make()` has a
zero-GPU-cost fast path: when a candidate Anchor's footprint Region is
*already* Resident (attached to some other Anchor already), it just does

```cpp
dependents_[region].insert(anchor_id);
dependencies_[anchor_id].insert(region);
```

-- free to do, and exactly how region-sharing accumulates in practice. Once
a Region has 2+ dependents, the old rule made it permanently unreclaimable
(every single dependent Anchor would need to be evicted in the same instant
for it to ever count), so budget-constrained runs quickly filled up with
Regions nobody could ever reclaim, and every subsequent promotion attempt
hit "no eligible victim" and gave up.

## Design (agreed after extensive discussion, see conversation history for
the back-and-forth)

**Group** = a set of Anchors that get evicted together. Formed by
overlap-based clustering: a newly-promoted Anchor joins the *existing*
group whose Region footprint it overlaps by >= `group_merge_overlap_threshold`
(if that group has room under `max_eviction_group_size`); otherwise it
starts a new singleton group. Groups never merge, only grow or dissolve.
This is explicitly a **clustering hint layered on top of `dependents_`/
`dependencies_`** (the sole ground truth for correctness) -- staleness in
the hint can only make eviction *less* effective, never unsafe.

Key pieces (`region_manager.{hpp,cpp}`, `replacement_policy.{hpp,cpp}`):

- `CoordinatorConfig::group_merge_overlap_threshold` (default 0.5) /
  `::max_eviction_group_size` (default 1 -- backward-compatible no-op,
  reproduces the original sole-ownership rule exactly).
- `RegionManager::assignAnchorToGroup()` -- the clustering logic above.
- `EvictionCandidate::group_members` -- every anchor that must be evicted
  together with this one. `buildEvictionCandidates()` still emits one entry
  per anchor (so legacy policies' by-anchor-id lookups keep working), but
  `reclaimable_bytes` is now computed as "this Region's entire `dependents_`
  set is a subset of this anchor's group" instead of "== 1".
- `CostAwareReplacementPolicy::groupRetentionDensity()` -- worst (hottest,
  most conservative) density across a candidate's `group_members`, used
  wherever the policy used to look up a single anchor's density.
- `RegionManager::buildRelocationPlan()`'s victim-selection loop -- when a
  policy picks a victim anchor, the *whole* group tied to that anchor's
  `EvictionCandidate::group_members` goes into `plan.evictions` together in
  one step, then all of those anchors' entries are removed from the
  candidate pool before the next iteration (so no group member is offered
  again as a separate, redundant selection).
- RoutingCache/dependency cleanup needed **no new code**:
  `RegionManager::retireAnchorsNow()` already walks its input anchor list
  and calls `forget()` + `replacement_policy_->onAnchorEvicted()` +
  `routing_cache_->erase()` per anchor -- group eviction just calls this
  with the whole group's member list instead of a singleton.

## Verification

- Two new targeted unit tests in `region_manager_coordinator_test.cpp`:
  `GroupEvictionReclaimsRegionSharedByMultipleAnchorsWhenCapAllowsIt` (proves
  a shared Region *does* get reclaimed, evicting both dependents together,
  when `max_eviction_group_size` allows it) and
  `SharedRegionStaysUnreclaimableWithDefaultGroupCap` (proves the default
  config's behavior is unchanged -- a regression guard for the "no behavior
  change unless explicitly raised" backward-compat claim). Both pass, log
  output matches the intended mechanism exactly.
- Full suite: 356/357 passing after the change (the 1 failure is
  `HnswlibIndexInsertAfterPromotionTest.DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot`,
  pre-existing and unrelated -- a documented Region-staleness issue from
  earlier in this investigation, not touched by this work).
- Direct log analysis at 100K-scale with `group_merge_overlap_threshold=0.1`
  showed groups actually forming and growing to the configured cap (sizes 1
  through 10 all observed), and the "no eligible victim found" share of
  rejections dropping from 97.4% to ~44%.

## Outcome: mechanism works, end-to-end benchmark barely moved

Sweep at 100K base / 15 steps / 10MiB budget (~30% of estimated full
residency), comparing `max_eviction_group_size=1` (baseline, original
sole-ownership rule) vs `=10`:

| config | candidates_rejected | insert ms/op | search ms/op |
|---|---|---|---|
| serial (group_size=1) | 261,989 | 3.0524 | 0.9853 |
| serial_with_group_eviction (group_size=10) | 262,217 | 2.9899 | 0.9351 |
| batched_with_wait (group_size=1) | 262,558 | 2.6436 | 0.8835 |
| batched_with_wait_and_group_eviction (group_size=10) | 262,316 | 2.7200 | 0.9644 |

Virtually unchanged, despite the mechanism itself being verified correct.

**Why**: fixing the "no eligible victim" bottleneck didn't remove
rejections, it just let more candidates reach the *next* gate --
`CostAwareReplacementPolicy::evaluateAdmission()`'s hysteresis check
(`candidate_density >= best_victim_density * admission_hysteresis`, default
`admission_hysteresis = 1.0`). Reject-reason share shifted from
97.4% "no eligible victim" / 2.6% hysteresis to roughly 44% / 56%. Total
reject count stayed about the same because the second gate absorbed almost
exactly what the first gate stopped blocking.

See `replacement_policy.cpp:632-686` for `evaluateAdmission()`. Two
open questions flagged but not yet resolved:

1. Whether `admission_hysteresis=1.0` (require the incoming candidate to be
   *at least as valuable* as the best victim, no slack at all) is simply too
   strict for this traffic shape, and would look different at a lower value.
2. Whether `candidate_density` (line ~672-677: raw, undecayed observation
   count / rounded incremental-byte units, normalized by
   `allocation_unit_bytes` *twice*) and `best_victim_density` (via
   `victimRetentionDensity()`, line 606-614: decayed heat / byte count
   rounded to a **1-byte** unit, i.e. `allocation_unit_bytes` never enters
   at all) are actually on comparable scales. Confirmed via `git log` that
   this exact formula predates this session's group-eviction work entirely
   (introduced in commit `fbf4126`) -- group eviction didn't introduce this,
   it just stopped a different, bigger bottleneck from hiding it.

Both are the subject of the next entry in this report, generalized into a
6-way replacement-policy comparison rather than tuning `CostAwareReplacementPolicy`
in isolation -- see
[2026-08-27-replacement-policy-sweep.md](2026-08-27-replacement-policy-sweep.md).
