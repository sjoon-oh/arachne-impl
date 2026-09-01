# Why pass count varies: the per-pass budget cap, and timing-sensitive backlog accumulation

Follow-up to [2026-09-01-eviction-cache-exclusion-set-fix.md](2026-09-01-eviction-cache-exclusion-set-fix.md)
(entry 12), which left open why `processRelocationBatch` count is still
elevated at 10M scale (316 in that entry's own re-run, vs 16-97 ever seen at
1M) even after fixing the victim-selection copy. This entry investigates
that directly, tracing-only (no decision/admission/eviction logic touched).

## Method

Two additions, both diagnostic-only:

1. `processRelocationBatch()` split into `_normal`/`_forced` trace-scope
   markers by `retain_failed_candidates` (`coordinatorLoop()` passes this as
   exactly `!forced && !stop`), to separate ordinary `trigger_interval`-paced
   passes from a forced/stop drain's own back-to-back do-while loop.
2. A new per-*pass* (not per-candidate) `ARACHNE_LOG_INFO` at each of
   `buildRelocationPlan_collect`'s three possible loop-exit points
   (`queue_drained`, `defer`, `over_budget`, `max_promotion_bytes_per_pass`),
   logging `examined` (candidates the collect loop actually looked at this
   pass) and `admitted` (`plan.promotions.size()`).

Re-ran the same 10M/50-step/single-thread/cost_aware/256MB-budget config
entries 11-12 used. One run kept `--quiet-logs` (matching prior methodology
exactly); a second dropped it (needed to actually see the new per-pass
summary lines, piped through a `grep` filter before ever touching disk so
volume stayed manageable -- entry 11's Finding 2 already established
`fmt::format()`'s own cost is negligible, 0.72s across 1M+ calls, so this
was expected to be safe).

## Finding 1: pass count varies 15x across identical back-to-back reruns of the fixed binary

| run | total passes | forced | normal | post-loop gap |
|---|---|---|---|---|
| entry 12's re-run | 316 | -- | -- | 7s |
| this entry, run 1 (quiet) | 2,228 | 2,151 | 77 | 954s (15.9 min) |
| this entry, run 2 (verbose) | 149 | -- | -- | ~0s |

Same binary (post-fix), same config, same index, same workload. This is a
**15x spread in total pass count from three back-to-back runs of nothing
that changed**. This alone says pass count was never going to be a clean,
deterministic function of "scale" -- entry 11's "44-250x explosion" framing
compared one 10M run against a *range* (16-97) from several different 1M
configs; a fairer comparison is range-to-range, and the 10M range overlaps
much more than that framing suggested once you see 149-2,228 from identical
reruns.

## Finding 2: during active streaming, the dominant pass-ending reason is the per-pass budget cap, not queue exhaustion

Every one of the first 116 passes in the verbose run ended with
`reason=over_budget` -- `plan.promotions`' cumulative bytes (summed across
*only this pass's own newly-collected candidates*, not existing residents)
crossed the full 256MB budget, so the collect loop popped the one candidate
that pushed it over, requeued that single candidate, and broke -- long
before the policy's own pending queue was anywhere near empty. Typical
`admitted` per over-budget pass: 979-4,239 (median 1,452) -- i.e. **each
pass that hits this cap replaces roughly the GPU's entire resident
population**, since 256MB / ~1,450 candidates implies ~177KB/candidate,
consistent with entry 11's own ~1,954-region-at-this-budget estimate.

This cap is **budget-driven, not queue-driven**: as long as the pending
queue has at least ~1,300+ eligible candidates waiting, a pass can only ever
admit up to what the *fixed* 256MB budget allows, no matter how much larger
the backlog actually is. A 10x-bigger corpus (10M vs 1M) generating a
correspondingly larger total *volume* of promotion requests over the run,
against the *same absolute* GPU budget, mechanically needs proportionally
more such budget-capped passes to work through -- the budget doesn't scale
with the corpus, so the same fixed per-pass ceiling has to be paid more
times.

## Finding 3: near the end of the run, most passes admit almost nothing despite examining thousands of candidates

The last ~30 passes in the verbose run shifted character: `reason=
queue_drained` (the loop finally *did* run the queue dry, not hit the
budget cap), `examined` in the thousands (up to 241,262 in one pass) but
`admitted` frequently **0** -- e.g. `examined=4385 admitted=0`, `examined=
5550 admitted=177`, `examined=4752 admitted=0`. Once GPU residency
approaches a reasonably "settled" state, CostAware's hysteresis check
(`candidate_density >= best_victim_density * admission_hysteresis`) rejects
the overwhelming majority of whatever's left in the backlog -- these are
candidates that were requested but never got serviced while the stream was
still running, so by the time they're finally looked at they're competing
against a resident set that's had a chance to actually accumulate real
heat. Rejects don't stop the collect loop (only queue-empty, defer, or the
budget cap do), so a pass in this regime pays the O(1)-ish per-candidate
admission-scan cost thousands of times over while accomplishing almost no
actual promotion.

## Finding 4: pass count is itself sensitive to how fast the Coordinator gets serviced -- confirmed directly, not just inferred

The quiet run (2,228 passes, 2,151 of them in the forced post-loop drain,
954s gap) and the verbose run (149 passes, ~0s gap) used the *identical*
binary and config -- the only difference was `--quiet-logs`, which changes
nothing about admission/eviction decisions, only how much I/O each pass
does. Total wall clock barely differed (308s vs entry 12's 297-337s
range), but the verbose run's passes were **far larger on average**
(admitting/examining thousands per pass) and there were **15x fewer of
them**. The mechanism: how much backlog accumulates in the policy's pending
queue before the Coordinator next gets to service it depends on the
relative pace of (a) candidates arriving vs (b) how quickly each service
opportunity actually runs -- a slower per-pass cost (whether from I/O, as
demonstrated here, or from the entry-11/12 victim-selection copy) delays
the next service point, letting more candidates from the *same* fixed
arrival stream pile up into a *larger* next batch. This is exactly why
entry 12 saw pass count itself drop (4,072 -> 316) purely from fixing an
unrelated per-pass cost: it was never really "fewer passes needed" in a
structural sense -- it's the same fixed backlog getting serviced in
whatever-size chunks the Coordinator's current pace happens to produce.

## Finding 5: total work stays roughly stable across that 15x pass-count swing -- pass count is a chunking artifact, not a cost signal

If pass count is timing-sensitive chunking rather than a proxy for total
work, the total number of candidates actually *examined* across a whole
run's passes (summed, not per-pass) should stay roughly constant even while
pass count itself swings 15x. It does:

| run | total passes | total candidates examined |
|---|---|---|
| entry 12's re-run (fix-verify) | 316 | ~1,018,002 |
| this entry, run 1 (quiet) | 2,228 | 1,146,796 |
| this entry, run 2 (verbose) | 149 | ~932,237 (180,976 over_budget + 751,261 queue_drained) |

**149 to 2,228 passes (15x) against 932K to 1.15M candidates examined (1.2x)**
-- the total amount of work done is stable to well within this
investigation's already-documented run-to-run noise floor, while the number
of passes it got split into varies by an order of magnitude. This makes
sense once framed as chunking: whether the Coordinator services a fixed
backlog as 149 large gulps or 2,228 small sips, each candidate still gets
looked at (roughly) once either way -- what changes is only how many
separate `buildRelocationPlan()` invocations that examining gets divided
across. (Some amplification exists -- the `over_budget` requeue path and a
normal pass's "give up" requeue both cause a requeued candidate to be
examined again later -- but it's evidently small relative to the total,
not the source of the 15x pass-count spread.)

The practical implication: **pass count was never a reliable signal for
this system's actual cost**, before or after entry 12's fix. The real cost
driver is total candidates examined/admitted (which tracks workload volume
sensibly) and the *per-examination* cost (which entry 12 fixed, at the
`buildRelocationPlan_evict` call site). Entry 11's framing treated a
symptom (many passes) as if it revealed the disease; the actual disease
(entry 11/12's copy) manifested as expensive individual passes, and pass
*count* just happened to correlate with it by coincidence of this
particular investigation's specific runs, not by any causal mechanism worth
chasing further.

## Synthesis: is this "resolved"?

Not with a single fix, and that's the honest conclusion -- and, per Finding
5, not really a thing that needed a fix in the first place. Pass count at
any scale is `(total candidate volume over the run) / (candidates serviced
per pass)`, where the numerator scales with corpus/workload size and the
denominator is capped by a *fixed* GPU budget (mechanically forcing more
passes as corpus grows) but also depends on incidental Coordinator-service
timing (demonstrated to swing 15x on its own, independent of scale). Entry
11's framing -- a single root cause producing a clean 44-250x multiplier --
doesn't hold up under closer inspection; the real picture is two structural
effects (fixed budget vs. growing candidate volume; hysteresis rejecting
most of a late backlog for free), a timing-sensitivity effect that makes
any single run's exact pass count more noise than signal, and -- the
clarifying result -- a total-work figure underneath all of that noise which
stays stable (Finding 5). This matches this investigation's own entry 3
precedent (a documented 5.8x reproducibility spread for a different metric)
-- pass count joins that list as a noisy-but-harmless metric, not a new
performance problem to chase.

This reframes what "workload characteristic vs. Arachne framework problem"
means for this specific question: the *structural* drivers (fixed budget
vs. growing corpus, hysteresis needing to evaluate before it can reject)
are inherent to any bounded-capacity replacement system facing a streaming
workload larger than that capacity -- not an Arachne-specific defect, the
same tension any fixed-size cache has against a growing corpus. The
*timing-sensitivity* piece (how that structural work gets chunked into
passes) is Coordinator-specific, but Finding 5 shows it doesn't change how
much work gets done, only how it's divided up -- so it doesn't rise to
"framework problem" either, just an implementation detail worth knowing
about before ever using pass count as a health metric again.

## What changed (tracing-only)

- `RegionManager::processRelocationBatch()`: `_normal`/`_forced` marker
  scopes (zero-duration, count/start_ns only -- see the scope's own code
  comment for why duration_ns from these two is meaningless).
- `RegionManager::buildRelocationPlan()`'s collect loop: one new
  `ARACHNE_LOG_INFO` per pass-ending branch (4 sites), reporting
  `examined`/`admitted`/reason. Existing per-candidate log and every
  decision/admission/eviction code path untouched.
- 374/375 suite passing (same pre-existing unrelated failure); `build/`
  rebuilt and re-verified after these changes, no regression (expected --
  logging-only additions on already-existing branches).

## Trace/log locations (not copied into git)

Both runs' full logs and CSVs were deleted after the numbers above were
extracted (~1.4-1.5GB each, same disk-hygiene convention as every prior
entry). The verbose run's log is the source for every `examined=`/
`admitted=`/`reason=` figure cited above.
