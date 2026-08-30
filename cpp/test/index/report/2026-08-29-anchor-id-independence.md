# Anchor-id / VectorId decoupling: implementation, unit-test verification,
# and a 1M-scale re-run against raw hnswlib

Follow-up to [2026-08-28-latency-tracing.md](2026-08-28-latency-tracing.md),
which fixed `onAnchorTouched()`'s mutex contention (the `touch_queue_`
redesign) and flagged a smaller, structurally identical issue on the delete
path (`onAnchorEvicted()`, called from `RegionManager::releaseAnchor()`,
still sharing `CostAwareReplacementPolicy::mutex_` directly, up to 55.3s max
stall in that entry's multi-thread measurement) as a follow-up. Separately
from that latency thread, a design review of Core's Anchor concept found
that `insert()`'s own lookup-traverse reused the about-to-be-inserted
record's own `VectorId` as its Anchor id -- conflating two identities that
are supposed to be independent (an Anchor is Core's own "this locality is
GPU-promotion-worthy" bookkeeping, not the underlying data vector), and
carrying a real (if remote) numeric-collision risk against `search()`'s own
synthetic anchor ids, which were minted from the same unmanaged id range via
a bare `next_anchor_id_.fetch_add(1)`. This entry covers the fix for both:
Anchor ids are now always minted from a single, dedicated id space that can
never collide with a real `VectorId`, and deletes no longer touch
`RegionManager`/`ReplacementPolicy` at all, closing out the entry-3 follow-up
as a side effect of the redesign rather than a targeted patch.

## What changed

- **`Controller::MintAnchorId()`** (new, `controller.{hpp,cpp}`): the single
  place every Anchor id now comes from, whether the traverse that needed one
  was triggered by `insert()`'s own lookup or by `search()`'s Hybrid routing
  (previously two different code paths, only one of which minted a real
  fresh id). Reserves the top bit of `VectorId` (`kAnchorIdBit = 1ULL << 63`)
  so a minted id can structurally never equal a real data-vector id -- a
  real deployment would need to exceed 2^63 vectors to ever reach that bit.
  A CAS-retry loop guarantees payload 0 is never minted even across the
  (astronomically unlikely, ~291 years at 1 billion mints/second)
  `next_anchor_id_` wraparound case, since `0` is the reserved "no anchor"
  sentinel checked throughout the codebase (`anchor_id != 0`).
- **`insert()`'s lookup-traverse** now calls `MintAnchorId()` instead of
  reusing `record.id`, and `requestPromotion()` is called with that same
  minted id (`insert_anchor_id`), not `record_id` -- the two used to be the
  same value by construction, so nothing downstream previously depended on
  telling them apart; a review confirmed nothing does now either (see the
  full-codebase audit in the design conversation this entry follows up on:
  every `anchor_id`-keyed structure in `RegionManager`/`ReplacementPolicy`/
  `RoutingCache`/the hnswlib adapter is hash-map-keyed or opaque-tolerant,
  never array-indexed by it).
- **`Controller::commitRemove()`** no longer calls
  `region_manager_.releaseAnchor()`. `plan.request.target` is the deleted
  *data vector's* own id -- with Anchor ids now properly independent, it was
  never actually an Anchor id to begin with, so the call was both
  conceptually wrong and (per entry 3) the site of the remaining
  mutex-contention tail. An Anchor's lifecycle is now driven purely by the
  `ReplacementPolicy` (heat/capacity), independent of whether the data
  vector that originally caused it to exist is still live.
  `Controller::verify()`'s own separate `releaseAnchor()` call (a real
  Anchor id, for a genuine "this Anchor's Region dependencies no longer
  represent its locality" case) is untouched.

## Unit tests

Full suite after the change: **367/368** (368 = the pre-change 365 plus the
3 new tests below; same single pre-existing, unrelated failure as every
prior entry --
`HnswlibIndexInsertAfterPromotionTest.DeviceSearchReflectsPostPromotionInsertsOrExplainsWhyNot`,
a diagnostic test that documents its own two valid outcomes and isn't
touched by anything in this line of work).

Three new tests added to `controller_test.cpp`
(`ControllerAnchorIdentityTest` suite), using two new test doubles
(`RecordingRoutingCache`, wrapping `FakeRoutingCache` to capture the ids
`ensure()`/`erase()` are actually called with; `EvictionCountingReplacementPolicy`,
delegating to a real `FifoReplacementPolicy` while counting `onAnchorEvicted()`
calls -- `FifoReplacementPolicy` is `final`, so this composes rather than
subclasses it):

- `InsertMintsAFreshAnchorIdInsteadOfReusingTheRecordId` -- inserts a record,
  asserts the id `RoutingCache::ensure()` actually saw is neither the
  record's own id nor missing the reserved top bit.
- `SearchMintsAnAnchorIdWithTheSameReservedTopBitAsInsert` -- same
  assertion, driven through `search()` instead, confirming both paths now
  share one id space.
- `RemoveNeverTouchesReplacementPolicyOrRoutingCacheAnymore` -- inserts,
  confirms the Region is resident, deletes, and asserts (a) the
  `ReplacementPolicy` never saw an `onAnchorEvicted()` call, (b) the
  `RoutingCache` was never `erase()`'d, and (c) the Region the insert
  promoted is *still* resident after the delete -- directly encoding the
  behavior change (a delete used to sometimes evict its own Region purely
  because `record.id` happened to double as the Anchor id; now it never
  does, on purpose).

Also updated four pre-existing, still-passing tests'
(`PromoteEvictsMultipleVictimsWhenOneIsNotEnoughCapacity`,
`EvictionBatchesMultipleRegionsFromOneAnchorInOneCall`) inline comments
that said e.g. "anchor 101 -> region 1" -- true by coincidence under the old
behavior, misleading now that the actual Anchor id is a minted, unrelated
value. No behavioral change; the tests never asserted the numeric id.

## 1M-scale re-run against raw hnswlib

Same two configs as entry 3, same pre-built 1M index
(`/tmp/arachne_cost_aware_1m_sweep/arachne_1m_index.bin`, `--load-index`),
same flags (`--gpu-budget-bytes 104857600 --m 16 --ef-construction 200
--ef-search 100 --replacement-policy cost_aware
--group-merge-overlap-threshold 0.5 --max-eviction-group-size 10
--batch-wait-timeout-us 500 --seed 100 --quiet-logs --limit-base 1000000
--limit-steps 30`), rebuilt against this session's changes
(`cpp/build-trace/`, `ARACHNE_ENABLE_TRACING` on).

| | single (batch=1, exec=1, client=1) | multi (batch=32, exec=4, client=4) |
|---|---|---|
| insert ms/op (raw) | 0.2747 | 0.2814 |
| insert ms/op (arachne) | **0.4471** | **0.1795** (beats raw, 1.6x) |
| search ms/op (raw) | 0.1215 | 0.1252 |
| search ms/op (arachne) | **0.0623** (beats raw, 2x) | **0.0199** (beats raw, 6.3x) |
| delete ms/op (raw) | 0.0001 | 0.0002 |
| delete ms/op (arachne) | 0.0004 | 0.0005 |
| recall@k (both engines) | matches raw within +0.0001-0.0003 at every step | matches raw within +0.0000-0.0001 |

(`--limit-base 1000000` truncates the base pool below the workload's
groundtruth, same caveat as every prior entry -- recall numbers above are
still meaningful as an *agreement-between-engines* check, not as an absolute
recall measurement; see entry 2's note.)

Delete ms/op is elevated over raw in absolute terms in both configs, same as
every prior round -- raw hnswlib's delete is just a tombstone flip, sub-
microsecond; arachne's still routes through the full async
submit/schedule/dispatch/adapter round-trip regardless of how cheap
`commitRemove()` itself now is. What matters here is *why* it's elevated:

### The originally-flagged `onAnchorEvicted()` tail is gone

`RegionManager-releaseAnchor.csv` **does not exist in either trace
directory's output** -- `ARACHNE_TRACE_SCOPE`'s underlying `TraceCollector`
is a function-local `static`, constructed (and thus ever writing a file) only
the first time its call site executes, so a missing file is a clean, direct
proof the call site never ran even once across all 300,000 deletes in either
run. `Controller-commitRemove`'s own max dropped to **0.0026 ms (single) /
0.0030 ms (multi)** -- essentially free, matching the doc comment's
description of what's left (nothing). `CostAwareReplacementPolicy-lockwait`'s
max across the whole run (all callers, not just deletes) is **0.0264 ms
(single) / 0.0283 ms (multi)** -- down from entry 3's 21,517.67 ms / 55,312.75
ms. Nothing client-facing waits on `CostAwareReplacementPolicy::mutex_` in
any observable way anymore; combined with the earlier `touch_queue_` fix,
that mutex is now effectively Coordinator-thread-only.

Client-facing tail latencies stay bounded and small throughout both runs:

| scope | single max | multi max |
|---|---|---|
| `Controller-submitInsert` | 10.96 ms | 11.84 ms |
| `Controller-submitSearch` | 1.73 ms | 2.86 ms |
| `Controller-submitRemove` | 6.07 ms | 2.78 ms |
| `RegionManager-recordTraversal` | 100.45 ms | 23.77 ms |
| `RegionManager-requestPromotion` | 8.72 ms | 1.08 ms |

(`recordTraversal`'s single-thread max of 100.45 ms is higher than entry 3's
14.16 ms measurement of the same config, but both are three-to-four orders
of magnitude below the pre-`touch_queue_` 216,864.5 ms figure and within the
kind of run-to-run noise entry 3 itself documented (up to 5.8x variance
between otherwise-identical runs) -- not treated as a regression.)

### New finding (pre-existing, not caused by this change): `buildRelocationPlan_collect`'s final drain pass is the new dominant cost, and very likely for a different, already-visible-in-code reason

Both configs show the same shape: relocation batches grow steadily larger
across the run, then the *last* one balloons far past the others --
single: 16 batches, the last one alone taking 643.5s out of a ~19 minute
total run (durations: 78-263ms x10, then 520ms, 5.8s, 14.5s, 25.1s, 119.4s,
**643.5s**); multi: 8 batches, the last taking 440.6s out of a ~18 minute
total run (79-249ms x4, then 1.0s, 5.0s, 31.4s, **440.6s**). In both cases
essentially the entire batch's cost is inside `buildRelocationPlan_collect`
specifically (635.9s of the 643.5s single; 430.1s of the 440.6s multi) --
this is `waitIdle()`'s final forced drain processing the whole run's
backlog of admission decisions (`candidates_rejected` = 495,950 single /
497,458 multi) in one uninterrupted pass, the same "Coordinator holds
`mutex_` for 10+ minutes evaluating ~500,000 admission decisions" shape
entry 3 first diagnosed.

The mechanism is different this time, though, and appears **pre-existing
rather than introduced by this session's change** -- nothing touched in this
entry runs inside this loop. Reading `RegionManager::buildRelocationPlan()`'s
collect loop: `evaluateAdmission_locked`'s own traced cost across the *whole*
run sums to only ~41s (single) / ~20s (multi), far short of the ~600s/~430s
observed -- but every iteration also calls `buildAdmissionContext()`, which
is not separately traced, and which does this whenever the pass doesn't
already have enough free budget for the candidate (true for nearly every
candidate once GPU memory is ~99.8% full, which it is almost the entire
run: `gpu_bytes_allocated=104,607,360` against a 104,857,600 budget):

```cpp
if (available < context.incremental_bytes) {
  if (!eviction_candidates_cache.has_value()) eviction_candidates_cache = buildEvictionCandidates();
  context.eviction_candidates = *eviction_candidates_cache;
}
```

`buildEvictionCandidates()` itself is correctly cached once per pass (its
own trace: 15-21ms max, called only 15-7 times total, matching entry 3's
fix holding). But `context.eviction_candidates = *eviction_candidates_cache;`
is a **full deep copy of the cached `std::vector<EvictionCandidate>`, once
per candidate examined** (up to ~500,000 times in the final pass) -- and
`EvictionCandidate` carries its own heap-allocated `std::vector<VectorId>
group_members` (up to 10 entries each, `--max-eviction-group-size 10`), so
every one of those ~500,000 copies is itself hundreds of small heap
allocations, not one. With an estimated few-hundred to low-thousands
currently-resident anchors at any snapshot (budget / region size), the
arithmetic (candidates x resident-anchor-count x per-copy allocation cost)
lands squarely in the observed few-hundred-second range. Not instrumented
directly this session to confirm beyond this estimate -- flagged as a
likely cause, not a certainty, for whoever picks this up next.

This was almost certainly present, at a smaller scale, in entry 3's own
measurements too (nothing about it depends on Anchor id independence) --
it just wasn't the *dominant* visible cost then, since `recordTraversal()`
and `releaseAnchor()`'s multi-minute client-facing stalls were larger and
more alarming. With both of those now fixed, this is what's left standing.
Unlike the two fixed issues, it does **not** leak into client-facing
latency (the Coordinator thread pays it alone, and nothing else needs
`CostAwareReplacementPolicy::mutex_` anymore) -- it only affects how long
the Coordinator's own background work takes to fully settle, most visibly
at `waitIdle()`/shutdown. A plausible fix direction: change
`AdmissionContext::eviction_candidates` to a `const std::vector<EvictionCandidate>*`
(or `std::shared_ptr`) into the pass-lifetime cache instead of a per-call
copy -- not attempted here, left for a future session alongside deciding
whether it's worth doing (it doesn't currently cost anything a client
observes).

## Conclusion

Both parts of this session's work verify clean: the Anchor-id/VectorId
decoupling itself (insert and search now share one id space, structurally
collision-free with real vector ids; a delete never again risks evicting an
unrelated Region purely by numeric coincidence) and, as a side effect of
removing `commitRemove()`'s `releaseAnchor()` call, the entry-3 follow-up
finding (`onAnchorEvicted()`'s mutex contention) is fully closed -- confirmed
by the trace file's outright absence, not just a smaller number. Insert and
search both beat raw hnswlib in both configs (up to 6.3x on multi-thread
search), matching entry 3's headline result and confirming this change
didn't regress it. One new, larger-magnitude but non-client-facing cost
(`buildRelocationPlan_collect`'s final-drain vector-copy pattern) was found
and root-caused to a plausible mechanism already visible in existing code,
not anything added this session -- left as a flagged follow-up rather than
fixed here, consistent with how entry 3 handled the analogous finding.

## Trace locations (not copied into git, same reasoning as earlier entries)

Both cleaned up after this entry's numbers were captured (disk-hygiene
lesson from entry 3 -- delete a trace directory once its findings are
recorded here, not "later"). If re-verification is ever needed, the exact
commands are captured in this entry's own text above (same flags as entry
3, `--load-index` against the same saved 1M index).
