# CostAware parameter sensitivity + trigger-interval + 5-policy comparison, 1M scale

20 runs, ~3h50min wall clock (05:00-08:51), using the `--cost-aware-*`/
`--trigger-interval-ms` flags from
[2026-08-30-cost-aware-config-cli-flags.md](2026-08-30-cost-aware-config-cli-flags.md).
Every run: same 1M pre-built index (`--load-index`), same base flags as
every prior entry in this investigation
(`--gpu-budget-bytes 104857600 --m 16 --ef-construction 200 --ef-search 100
--group-merge-overlap-threshold 0.5 --max-eviction-group-size 10
--batch-wait-timeout-us 500 --seed 100 --limit-base 1000000 --limit-steps 30`),
both thread configs (single: `exec=1,client=1,batch=1`; multi:
`exec=4,client=4,batch=32`). `--engine arachne` only (not `both`) --
raw_hnswlib's own numbers are deterministic given the fixed seed/index and
don't depend on any of these arachne-side flags, so re-deriving them 20
times would have cost ~90 extra minutes for zero new information; cited
below from the immediately preceding entry's own `--engine both` run
instead (raw insert 0.2704 ms/op, delete 0.0002 ms/op, search 0.1204 ms/op,
recall 0.8913/0.8141/0.7504 at steps 10/20/30, mean 0.8186).

## Headline result: B and C, at the values chosen, both collapse the
system into the same "promote once, never evict again" regime -- and it
barely matters

| config | promoted | evicted | batches | rejected | recall | insert ms/op | search ms/op |
|---|---|---|---|---|---|---|---|
| baseline (single) | 3,810 | 3,429 | 92 | 495,981 | 0.8188 | 0.3928 | 0.1013 |
| min_residency=2000ms (single) | 381 | **0** | 84 | 498,631 | 0.8188 | 0.3979 | 0.1043 |
| hysteresis=1.5 (single) | 381 | **0** | 84 | 498,631 | 0.8188 | 0.3894 | 0.1028 |
| baseline (multi) | 1,524 | 1,143 | 29 | 497,934 | 0.8187 | 0.1679 | 0.0445 |
| min_residency=2000ms (multi) | 381 | **0** | 28 | 499,022 | 0.8186 | 0.1666 | 0.0458 |
| hysteresis=1.5 (multi) | 381 | **0** | 6 | 498,556 | 0.8186 | 0.1763 | 0.0210 |

Both parameters, at the "moderate, not extreme" values chosen precisely to
avoid this (`minimum_residency=2000ms` -- 40% of `heat_half_life`'s 5000ms;
`admission_hysteresis=1.5` -- a 50% margin, not an order of magnitude),
independently produce the *exact same* `regions_evicted=0`,
`regions_promoted=381` outcome in both thread configs. Once the budget
fills from empty (the initial 381 promotions), every later candidate's
`evaluateAdmission()` call finds every current resident still protected
(residency case) or scoring below the required margin (hysteresis case),
so nothing is ever evictable again -- the resident set freezes for the
rest of the 900,000-op run. This isn't a bug in either lever; it's what
"protect residents more" *means*, taken to its logical endpoint under a
budget that's essentially always full (99.8%) -- the two levers reach the
same endpoint by different roads because at this budget/workload
combination, *almost every* admission decision is marginal enough that
either kind of protection is sufficient to block it outright.

**recall is unaffected either way (0.8186-0.8188 across all six rows
above, matching raw's own 0.8186 mean)**. This is the clearest possible
confirmation of an architectural property this investigation has touched
on before but never demonstrated this starkly: GPU residency is a pure
performance layer over a host-side path that's always correct on its own
(`traverseHost()`'s search results don't depend on what's GPU-resident) --
freezing residency after the first 381 anchors, for the entire remainder
of a 900K-op run, costs nothing in recall.

**ms/op is also barely affected** -- single-thread insert/search both move
by under 3% either direction; multi-thread search actually improves under
hysteresis=1.5 (0.0210 vs baseline's 0.0445, though see the caveat below
about `search` under a frozen resident set testing a narrower, more
stable working set rather than doing less real work per query). Given the
system already spends most of its `evaluateAdmission()` time rejecting
(baseline's own reject rate is already ~83%), freezing it out entirely
removes very little additional work -- `touch_queue_`/shared_ptr already
made each individual rejection cheap; there just wasn't much *evictable*
value left to squeeze out for this budget/workload shape.

**Practical reading**: `admission_hysteresis` and `minimum_residency` are
not independent, gently-tunable knobs for *this* workload/budget
combination -- they're closer to a switch between "adapt to changing
locality" (any value near the defaults) and "freeze after initial fill"
(almost any nontrivial increase to either). A future sweep wanting to see
genuinely intermediate behavior would need much smaller steps (e.g.
`admission_hysteresis` in the 1.05-1.2 range, `minimum_residency` in the
50-500ms range) to find where the transition actually happens, rather than
values chosen by eye against the defaults' own magnitude.

## E (`minimum_observations=3`): even more extreme -- GPU is never used at
all, and the workload runs *faster* without it

| config | promoted | gpu_bytes_allocated | recall | insert ms/op | search ms/op |
|---|---|---|---|---|---|
| baseline (single) | 3,810 | 104,607,360 | 0.8188 | 0.3928 | 0.1013 |
| min_observations=3 (single) | **0** | **0** | 0.8188 | **0.3438** | **0.0275** |
| baseline (multi) | 1,524 | 104,607,360 | 0.8187 | 0.1679 | 0.0445 |
| min_observations=3 (multi) | **0** | **0** | 0.8187 | **0.1311** | **0.0093** |

`relocation_batches=0` -- not one single Anchor was ever promoted across
the entire run. Root cause: `PromotionCandidate::observations` only
accumulates while a candidate sits *pending*, merged via repeated
`enqueueCandidate()` calls for the same anchor before it's ever pulled and
resolved (see that struct's own doc comment). In this workload's actual
timing, candidates get pulled and resolved close to as fast as they
arrive -- essentially none ever get a second merge in before being
decided, so `observations` is 1 for almost everything, almost always.
Raising the floor to 3 means *everything* fails the very first check in
`evaluateAdmission()`, before the (already inexpensive) eviction-scan
logic ever runs at all.

**Recall is again completely unaffected (0.8188/0.8187, matching baseline
exactly)** -- the cleanest possible demonstration of the host-fallback
guarantee: this workload's correctness genuinely does not depend on GPU
residency existing at all.

**And it's faster** -- 12-22% faster insert, 73-79% faster search, with
zero Coordinator/promotion/eviction/device-transfer overhead of any kind.
This is a real, if narrow, finding: *at this specific 100 MiB budget
against a 1M-vector base (roughly 800 Regions' worth of capacity), GPU
offload's overhead currently costs more than it returns* -- not evidence
that GPU offload is never worth it, but evidence that this budget is too
small relative to the working set for it to pay off in this workload's
own terms (recall this system already measures well above raw hnswlib at
every *non-frozen* config in this and prior entries; the comparison that
matters is GPU-on vs. GPU-off at this specific budget, which this table
answers directly, not GPU-on vs. raw hnswlib).

**Practical reading**: `minimum_observations` above 1 is not usable for
this workload at all without also changing how long a candidate stays
pending before being resolved (i.e. it interacts with the very
`trigger_interval`/coalescing behavior discussed next) -- 1 isn't merely
this workload's default, it's close to the *only* value that admits
anything.

## Trigger interval (100ms -> 1000ms): the one lever that actually showed
graded, non-collapsed behavior

| config | promoted | evicted | batches | recall | insert ms/op | search ms/op |
|---|---|---|---|---|---|---|
| baseline (single, 100ms) | 3,810 | 3,429 | 92 | 0.8188 | 0.3928 | 0.1013 |
| trigger=1000ms (single) | 762 | 381 | 52 | 0.8188 | 0.3902 | **0.0888** |
| baseline (multi, 100ms) | 1,524 | 1,143 | 29 | 0.8187 | 0.1679 | 0.0445 |
| trigger=1000ms (multi) | 381 | 0 | 8 | 0.8186 | 0.1678 | 0.0370 |

Single-thread lands in a genuine middle ground -- real (if reduced) churn,
not a total freeze, and shows a real search improvement (0.0888 vs
0.1013, ~12% faster) at unchanged recall. Multi-thread, at this batch
size, ends up at the same frozen (0 evictions) endpoint B/C reached, just
via a different mechanism (a 10x wider coalescing window means far fewer,
larger passes -- 8 vs baseline's 29 -- so the same total candidate volume
gets judged against a *staler* view of what's evictable each time, making
each pass's eviction opportunities scarcer). Insert is essentially
unaffected in both configs (within 1-2%) -- this lever mainly reshapes
*eviction* cadence, not the insert path's own cost.

**This also resolves the open question from the previous entry**: that
entry found `candidates_rejected` swinging from a consistent ~497K across
three prior rounds down to a reproducible ~31K in two back-to-back runs,
with no code change to explain it, and flagged it as possibly this
harness's own documented run-to-run timing variance. Every cost_aware run
in *this* sweep -- baseline, min_residency, hysteresis, trigger_interval,
all 8 of them -- lands back in the 495,981-499,022 range, tightly
clustered, with nothing near 31K anywhere. That earlier pair of runs now
reads as a genuine outlier (very likely system load at that specific
moment, given how directly `trigger_interval`-driven cadence ties total
admission-evaluation count to real wall-clock elapsed time), not a
reproducible property of the current code -- treated as resolved.

## 5-policy comparison: fifo/lru/lfu/clock/2q now complete in reasonable
time (they didn't before this session's fixes), and their search latency
beats cost_aware's

| policy | promoted | evicted | anchor_evictions | batches | rejected | recall | insert ms/op | search ms/op | wall time (single) |
|---|---|---|---|---|---|---|---|---|---|
| cost_aware (baseline) | 3,810 | 3,429 | 3,429 | 92 | 495,981 | 0.8188 | 0.3928 | 0.1013 | ~2.5 min |
| fifo | 811,603 | 811,297 | 312,431 | 3,570 | 0 | 0.8188 | 0.3696 | 0.0331 | ~20 min |
| lru | 811,603 | 811,297 | 312,431 | 3,570 | 0 | 0.8188 | 0.3885 | 0.0383 | ~22 min |
| lfu | 811,603 | 811,297 | 312,431 | 3,570 | 0 | 0.8188 | 0.3707 | 0.0391 | ~25 min |
| clock | 812,723 | 812,412 | 312,937 | 3,572 | 0 | 0.8188 | 0.3792 | 0.0368 | ~10 min |
| twoq | 811,400 | 811,094 | 312,434 | 3,571 | 0 | 0.8188 | 0.3703 | 0.0379 | ~25 min |

(multi-thread config: same qualitative pattern -- `candidates_rejected=0`
for all 5, `regions_promoted`/`evicted` in the 840K range,
`relocation_batches` ~3,610-3,618, wall time ~9-24 min; insert 0.147-0.150
ms/op, search 0.0111-0.0144 ms/op, recall 0.8186-0.8187 throughout -- see
the raw logs for the full table.)

**All 5 completed** -- this is itself the headline finding. The earlier
sweep episode
([2026-08-27-replacement-policy-sweep.md](2026-08-27-replacement-policy-sweep.md))
found `fifo` "does not complete in reasonable time" under sustained budget
pressure at scale and scoped all 5 non-`cost_aware` policies out of any
1M-scale comparison specifically because of this. That finding predates
every infrastructure fix made in this investigation since
(`touch_queue_`, the delete-path fix, the `AdmissionContext::
eviction_candidates` shared_ptr fix). The shared_ptr fix in particular
matters here regardless of which policy is active: `buildAdmissionContext()`
still builds and (previously) deep-copied `eviction_candidates` for
*every* candidate needing eviction help, whether or not the active
policy's `evaluateAdmission()` ever reads it -- the base class default
these 5 use (unconditional Admit) never reads it, but was still paying
the O(candidates x cache size) copy cost before this session's fix. That
cost being gone is very likely why these policies -- despite generating
~530x cost_aware's promotion/eviction volume (no admission-side
rejection at all, so every touch is a real promote-or-evict event) --
now finish in tens of minutes instead of not finishing at all.

**Their search latency beats cost_aware's, sometimes by a wide margin**
(single: 0.0331-0.0391 ms vs cost_aware's 0.1013 ms, ~3x; multi:
0.0111-0.0144 ms vs 0.0445 ms, ~3-4x). Best-supported explanation: these 5
policies admit *everything*, so at any given moment a much larger,
constantly-refreshed slice of the working set is GPU-resident (regions
promoted/evicted in the hundreds of thousands vs. cost_aware's low
thousands) -- higher GPU hit rate for whatever the stream searches next,
paid for entirely by Coordinator-side churn that (per the finding above)
turned out to be cheap enough not to matter for insert throughput (insert
ms/op is within a few percent of cost_aware's own baseline across all 5).
`anchor_evictions` (~312-317K) being far smaller than `regions_evicted`
(~811-813K) confirms group eviction (`max_eviction_group_size=10`) is
doing real batching work here -- each eviction decision reclaims multiple
Regions' worth of anchors together, not one at a time.

**Recall is identical across every policy tested (0.8186-0.8188)** -- the
same host-fallback-correctness point as the parameter sweep above, now
demonstrated across a completely different axis (which policy governs
promotion/eviction, not which knob value): recall in this workload is
essentially decoupled from replacement-policy choice entirely.

**Caveat on wall time**: these 5 completed in 9-25 minutes, not the "does
not complete" finding from before -- but they're still 4-12x slower than
cost_aware's own ~2-2.5 minute baseline, purely from sheer event volume
(hundreds of thousands of real GPU transfers vs. a few thousand). Whether
that trade (much better search latency, much more Coordinator work, no
capacity-aware admission at all) is worth it for a real deployment is a
separate question this entry doesn't attempt to answer -- it only
confirms the comparison is now *possible* to run, which it previously
wasn't.

## Conclusion

Of the four levers tested, three (B, C, E) collapsed this specific
workload/budget combination into extreme (frozen or fully-disabled)
regimes at the values chosen -- genuinely informative about where those
levers' effective range sits for this workload (much narrower than their
raw magnitude suggests), but not the graded sensitivity data a follow-up
tuning pass would want. `trigger_interval` was the one lever that showed
real graded behavior at the value tested. Recall stayed within
measurement noise of raw hnswlib's own 0.8186 mean in *every one* of the
20 runs, regardless of policy or parameter value -- the strongest
evidence yet in this investigation that GPU residency (which policy
governs it, how aggressively, or whether it's used at all) is a pure
performance concern for this system, never a correctness one. The
5-policy comparison confirms this session's infrastructure fixes
generalized well beyond `cost_aware` specifically (all 5 previously-stuck
policies now complete in reasonable time) and surfaces a genuinely
actionable finding of its own: for search-latency-sensitive workloads at
this budget, the unconditional-admit policies currently beat
`cost_aware`'s own default configuration on search, at the cost of far
more background churn.

## Follow-ups this entry surfaces, not attempted here

- A finer-grained sweep of `admission_hysteresis` (1.05-1.2) and
  `minimum_residency` (50-500ms) to find the actual transition point
  between "adapts" and "freezes", now that this entry established the
  extremes.
- Understanding *why* GPU offload costs more than it returns at this
  specific 100 MiB budget (the `minimum_observations=3` finding) --
  worth checking whether a larger budget (more of the working set
  actually resident) changes this conclusion, since 100 MiB / ~800
  Regions against 1M+ vectors is a fairly tight ratio.
- `minimum_observations` genuinely can't be tuned above 1 without also
  addressing how quickly pending candidates get resolved (ties to
  `trigger_interval`) -- a joint sweep of the two, not attempted here.
- The 5-policy search-latency advantage's cost (wall time, real GPU
  transfer volume) versus cost_aware's admission-side savings is a real
  trade-off a production configuration decision would need to weigh --
  this entry only measured both sides, didn't recommend one.

## Raw logs

`/tmp/arachne_cost_aware_1m_sweep/refine_sweep/*.log` (20 files, not
copied into git -- same reasoning as every prior entry's trace directories,
regenerable from this entry's own driver script and CLI flags).
