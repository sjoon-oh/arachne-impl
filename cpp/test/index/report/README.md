# Investigation reports

Cumulative notes for the `arachne_hnsw_workload_compare` investigation
(raw hnswlib vs. the full Arachne Controller stack, replayed against a real
SIFT1B streaming workload -- see the binary's own file-header comment and
`run_compare.sh` for how to run it).

Each file here is one investigation episode: what was tested, what was
observed, what we currently believe the cause is, and what's left open.
Entries are meant to be read in order -- later entries assume earlier ones as
background and link back to them rather than re-explaining. Add a new file
per episode rather than rewriting old ones; if an old conclusion turns out to
be wrong, say so in the new entry and correct the index below, but leave the
old file's text alone (it's a historical record of what we believed and why,
not a living doc).

## Index

1. [2026-08-27-group-eviction.md](2026-08-27-group-eviction.md) -- root-caused
   the "candidate rejected despite region being shared by multiple anchors"
   symptom to `RegionManager`'s original sole-ownership-only reclaimability
   rule; designed and implemented anchor-group-based eviction; verified the
   mechanism works via unit tests and log analysis; found via benchmark that
   fixing this bottleneck exposed a second one (hysteresis-based admission)
   that was previously hidden behind it, so end-to-end throughput didn't
   improve as much as the mechanism fix alone would suggest.
2. [2026-08-27-replacement-policy-sweep.md](2026-08-27-replacement-policy-sweep.md)
   -- compared all 6 `ReplacementPolicy` implementations at small scale
   (found the 5 non-`cost_aware` policies pay a much higher fixed per-touch
   cost, unrelated to eviction pressure, and scoped them out of the large-
   scale matrix), then ran a full 24-config `cost_aware`-only sweep at 1M
   base / 30 steps across batch size x thread count. Result: the original
   symptom is fully fixable, but only at specific (batch, threads)
   combinations -- 2 of 24 configs beat raw hnswlib's total time outright
   (0.84x, 0.92x), while the default/naive config (`batch=1, threads=1`)
   was the *worst* of all 24 (11.47x raw). **Caveat (see entry 3): this
   "beats raw" framing compares Arachne at up to 8 threads against
   `raw_hnswlib`, which always runs single-threaded -- not apples-to-apples.**
   See that file's final section for the full ranking, root-cause picture,
   and improvement directions.
3. [2026-08-28-latency-tracing.md](2026-08-28-latency-tracing.md) -- added
   real per-scope latency tracing (existing `ARACHNE_ENABLE_TRACING` build)
   to find out *where* time actually goes, since entry 2 only had aggregate
   counters. Corrected entry 2's "beats raw" framing (raw is always
   single-threaded). Found a serious reproducibility problem (the same
   config re-run gave up to 5.8x different numbers). Root-caused the actual
   bottleneck via direct measurement, confirmed across 3 independent runs:
   `RegionManager::recordTraversal()`'s latency is heavy-tailed (median tens
   of microseconds, but occasional multi-second-to-216-second stalls) because
   it contends for `CostAwareReplacementPolicy::mutex_` against the
   Coordinator, which can hold that mutex in a rapid-reacquisition "lock
   convoy" for 10+ minutes straight while evaluating ~500,000 admission
   decisions inside one `buildRelocationPlan_collect` pass. Implemented a fix
   (`ReplacementPolicy::shouldYieldPass()`, `CostAwareReplacementPolicy`
   yields after `max_pass_duration`, default 20ms; 6 new unit tests, 362/363
   suite passing) and verified it against the same two configs -- **mixed
   result, not a clean win**: worst-case mutex wait did drop substantially
   (6.6s -> 1.7s for the single-thread config), but forcing far more/shorter
   passes defeated `buildEvictionCandidates()`'s per-pass cache (an `O(all
   tracked anchors)` scan), ballooning that alone to 3.25 hours of redundant
   work across ~400,000 tiny passes and making the single-thread config's
   *total* wall-clock time far worse than before the fix. The multi-thread
   config regressed more mildly but still regressed (worst-case mutex wait
   actually got worse, 2.7s -> 12.6s). Root cause discussed further and
   correctly identified as `onAnchorTouched()` (called synchronously from
   worker threads) sharing `mutex_` with the Coordinator's admission/
   eviction scans -- fixed properly by moving `onAnchorTouched()` onto its
   own separately-locked MPSC queue (`touch_queue_`), drained lazily by the
   Coordinator wherever it already holds `mutex_`; `shouldYieldPass`'s
   default reverted to off (0 = unlimited) since this redesign removes what
   it was compensating for. **Confirmed fixed by direct measurement**:
   `recordTraversal()`'s worst case dropped from as bad as 216.9 seconds to
   7-14 milliseconds across both configs, insert/search now beat raw
   hnswlib even in the fair single-thread comparison (0.47/0.04 ms vs raw's
   1.89/0.77 ms), and the single-thread config's whole-run wall clock
   dropped from measured-in-hours back to ~22 minutes. 9 unit tests
   (2 new), 364/365 suite passing. Found one smaller, structurally identical
   follow-up issue on the delete path (`onAnchorEvicted()` still shares
   `mutex_`, now the dominant remaining tail at up to 55s) -- not fixed,
   flagged for a future session. Also hit and documented a real disk-space
   incident mid-investigation (accumulated trace directories filled `/tmp`
   to the point the harness itself couldn't write) -- recovered, with a
   process lesson recorded for next time.
4. [2026-08-29-anchor-id-independence.md](2026-08-29-anchor-id-independence.md)
   -- implemented Anchor-id/VectorId decoupling (`Controller::MintAnchorId()`,
   a single dedicated top-bit-reserved id space for every Anchor, replacing
   `insert()`'s old "reuse the record's own id" shortcut and unifying it with
   `search()`'s minting) and removed `commitRemove()`'s `releaseAnchor()`
   call (deletes no longer assume a data-vector id doubles as an Anchor id).
   3 new unit tests added (367/368 suite passing, same pre-existing unrelated
   failure as every entry above). Re-ran the same 1M-scale single/multi
   verification against raw hnswlib: **closes out entry 3's flagged
   `onAnchorEvicted()` follow-up outright** (its trace file no longer exists
   at all -- the call site never runs), insert/search still beat raw hnswlib
   in both configs (up to 6.3x), and client-facing tail latencies stay in
   the single-digit-to-low-double-digit milliseconds throughout. Found (but
   didn't fix) a new dominant cost in `buildRelocationPlan_collect`'s final
   backlog-drain pass -- root-caused to a likely pre-existing per-candidate
   `EvictionCandidate` vector-copy pattern in `buildAdmissionContext()`, not
   anything added this session, and confirmed **not** client-facing (the
   Coordinator thread pays it alone) -- flagged for a future session.
5. [2026-08-29-anchor-wraparound-guard.md](2026-08-29-anchor-wraparound-guard.md)
   -- closed the one concrete gap left open by entry 4: `MintAnchorId()`
   only guarded against wrapping onto the reserved 0 sentinel, not onto some
   other still-meaningful, previously-issued id. Added
   `RegionManager::isKnownAnchor()` and a sticky wrapped-flag so
   `MintAnchorId()` retries on an actual collision, at zero cost for every
   mint before a wraparound (which this entry also found is a `2^64`-mint
   event, not the `2^63` entry 4 said -- ~584.9 billion years at 1 billion
   mints/second, not ~291; entry 4's own number is left uncorrected in
   place as a historical record, per this README's own convention). 3 new
   unit tests, 370/371 suite passing (one new, investigated-and-dismissed
   flaky test unrelated to this change -- see the entry for why). 1M-scale
   re-run: no observable behavior change, `isKnownAnchor()`'s trace file is
   absent from both runs' output (the guard's fast path held for the entire
   run, as designed).
6. [2026-08-30-shouldyieldpass-removal.md](2026-08-30-shouldyieldpass-removal.md)
   -- removed `shouldYieldPass()` and everything that existed purely to
   support it (`CostAwareReplacementConfig::max_pass_duration`,
   `RelocationBatchContext`'s two pass-tracking fields, the collect-loop
   yield check, 6 tests), at the user's request now that the real fixes
   (`touch_queue_`, delete-path) make it unnecessary. Scoped narrowly --
   `RelocationBatchContext`'s other fields and `evaluateBatchAdmission()`
   untouched. 364/365 suite passing (same pre-existing failure); no 1M-scale
   re-run needed (removes an already-unused-by-default code path, zero
   runtime behavior change for any existing config).
7. [2026-08-30-eviction-candidates-shared-ptr.md](2026-08-30-eviction-candidates-shared-ptr.md)
   -- fixed entry 4's flagged (not yet fixed) finding: `AdmissionContext::
   eviction_candidates` changed from an owned `std::vector` (deep-copied
   once per *candidate*) to a `std::shared_ptr` (refcounted, shared once per
   *pass*) -- a plain reference was considered and rejected first (would
   dangle past `buildRelocationPlan()`'s own return, confirmed before
   implementing). `buildRelocationPlan_collect`'s worst pass dropped 68.1x
   (single) / 7.7x (multi); confirmed harmless to the other 5 policies
   (82/82 policy tests passing) and correct via a new test proving the
   *sharing* directly (same address across candidates), not just resulting
   values. 365/366 suite passing. **One real, honestly-reported side
   effect**: in the single-thread config only, the now-much-faster
   Coordinator's cadence shifted from 16 large passes to 97 small ones,
   interleaving its (unchanged-in-total) GPU work more finely with
   traversal execution on `--exec-threads 1`'s one shared worker --
   measured as a 1.9x stream-search wall-clock regression there (1.4x in
   multi-thread, still 4.8x faster than raw) -- not a correctness or recall
   issue in either config.
8. [2026-08-30-eviction-interface-unification.md](2026-08-30-eviction-interface-unification.md)
   -- collapsed `ReplacementPolicy`'s two eviction-selection entry points
   (`selectNextEvictionCandidate()`, pure-virtual; `selectEvictionCandidate()`,
   virtual-with-a-delegating-default) into one. **Correction made
   mid-investigation**: initially assumed the 5 non-`cost_aware` policies
   relied on that default -- re-reading the `.cpp` bodies found all 6
   already overrode `selectEvictionCandidate()` with real, independent logic
   (the same walk plus a `ContainsCandidate()` filter), so the "legacy"
   method was dead code nobody called, not a compatibility shim; this
   removes genuinely unused implementations, not a behavior-preserving
   rename. Fixed a stale `region_manager.hpp` doc comment describing a retry
   loop that doesn't match current code. 46 test call sites updated across 4
   files; 365/366 suite passing, 83/83 replacement-policy-filtered tests
   passing (5 unaffected policies confirmed). 1M-scale re-check: recall and
   ms/op both fine, but flags an **unresolved anomaly** --
   `candidates_rejected` came out ~31K in two runs here versus a consistent
   ~497K across three independent prior rounds, despite the admission/
   eviction decision code being provably byte-for-byte unchanged; likely
   this harness's own already-documented wall-clock timing variance
   (entry 3), not confirmed with a controlled A/B -- left open for whoever
   runs the next CostAware-tuning sweep to keep in mind.
9. [2026-08-30-cost-aware-config-cli-flags.md](2026-08-30-cost-aware-config-cli-flags.md)
   -- test-infra-only: exposed `CostAwareReplacementConfig`'s six tuning
   knobs (`minimum_observations`, `heat_half_life`, `minimum_residency`,
   `admission_hysteresis`, `potential_writeback_weight`,
   `maximum_incremental_bytes`) as `--cost-aware-*` CLI flags on
   `hnsw_workload_compare`, unblocking the CostAware-refinement sweep this
   entry's conversation planned (B/minimum_residency, C/admission_hysteresis,
   E/minimum_observations). No `--cost-aware-*` flag given is byte-for-byte
   the same as before (still the `nullptr` default-construction passthrough);
   `arachne_core` untouched. 365/366 suite passing (unaffected, same
   pre-existing failure). Smoke-tested, not yet swept.
10. [2026-08-30-cost-aware-refinement-sweep.md](2026-08-30-cost-aware-refinement-sweep.md)
    -- the sweep entry 9 unblocked, plus a new `--trigger-interval-ms` flag
    (`CoordinatorConfig::trigger_interval`, added alongside) and a 1M-scale
    5-policy comparison, 20 runs total (~3h50min). **Headline finding**:
    `minimum_residency=2000ms` and `admission_hysteresis=1.5` -- both chosen
    as moderate, non-extreme values -- independently collapse this
    workload/budget into the *identical* "promote once at startup, never
    evict again" regime (`regions_evicted=0` for the rest of the run, both
    thread configs); `minimum_observations=3` is even more extreme
    (`relocation_batches=0` -- GPU is never used *at all*). Recall stays
    within noise of raw hnswlib's 0.8186 mean in every one of the 20 runs
    regardless -- the clearest demonstration yet in this investigation that
    GPU residency (which policy, how aggressive, or on/off entirely) is a
    pure performance concern, never a correctness one, for this workload.
    `trigger_interval=1000ms` was the one lever showing genuinely graded
    (not collapsed) behavior. Also **resolves** entry 8's flagged, unresolved
    ~497K-vs-~31K `candidates_rejected` anomaly: every cost_aware run in this
    sweep (8 of them) lands back in the 495,981-499,022 range -- that
    earlier pair reads as a genuine one-off outlier now, not a reproducible
    property of the current code. **5-policy comparison**: fifo/lru/lfu/
    clock/2q all *complete* at 1M scale now (9-25 min each) -- entry 2 found
    they didn't, before this investigation's infrastructure fixes existed;
    their search latency beats cost_aware's default by ~3-4x, at the cost of
    ~530x the promotion/eviction volume (no admission-side filtering at
    all) and 4-12x cost_aware's own wall time.
11. [2026-08-31-10m-scale-coordinator-throughput.md](2026-08-31-10m-scale-coordinator-throughput.md)
    -- before attempting the requested 10M-scale sweep, calibration runs
    showed a severe, run-to-run-inconsistent gap (8-102 minutes) between a
    config's own printed op-time sum and its real wall clock. Root-caused
    via two new trace-only scopes (no logic touched): the streaming loop
    itself is never the problem (325s, matches its own timers exactly) --
    the entire gap is `RegionManager::shutdown()`'s forced final drain.
    Ruled out `ARACHNE_LOG_INFO`'s unconditional `fmt::format()` cost by
    direct measurement (0.72s total across 1M+ calls -- negligible).
    Found the real cause: `buildRelocationPlan()`'s victim-selection phase
    deep-copies `eviction_candidates_cache` on *every* pass needing
    eviction (`std::vector<EvictionCandidate> candidates = *eviction_candidates_cache;`)
    -- the same architectural pattern as entry 9's admission-side fix, at a
    *different* call site entry 9 explicitly reasoned was safe to leave
    alone ("called once per pass... never the cost the shared_ptr change
    targets"). That held at 1M scale (16-97 passes ever measured); at 10M
    scale this one run made **4,069** passes reaching the evict phase --
    turning a ~46-second total cost into 2,130 seconds (35.5 minutes), the
    dominant share of the observed gap. Confirmed not cost_aware-specific
    (the copy is policy-agnostic; the pre-entry evidence -- `fifo`'s own
    102-minute calibration run beating cost_aware's 75-minute one despite
    zero admission-scan cost -- already pointed this direction). Why pass
    count itself explodes 44-250x at 10M scale is flagged as a still-open
    sub-question. Fix direction proposed, not implemented (root-cause scope
    only). 365/366 suite passing (trace-only change, unaffected).
12. [2026-09-01-eviction-cache-exclusion-set-fix.md](2026-09-01-eviction-cache-exclusion-set-fix.md)
    -- implements and verifies entry 11's proposed (not attempted there) fix:
    `ReplacementPolicy::selectEvictionCandidate()` gained an `excluded_anchors`
    set parameter so `buildRelocationPlan()`'s victim-selection loop can share
    `eviction_candidates_cache` directly instead of deep-copying it every
    pass; all 6 policies updated to honor it. 9 new unit tests, 374/375
    suite passing (same pre-existing unrelated failure); existing
    multi-victim-per-pass integration tests (`StrictBatchEvictsEnough
    VictimsBeforePromotingWholeFootprint` etc.) confirm the exclusion set
    accumulates correctly across a pass's while-loop iterations. Re-ran
    entry 11's exact 10M config: the 68-minute post-loop gap dropped to 7
    seconds (~583x), `buildRelocationPlan_evict` itself dropped ~750x
    (2,130.0s -> 2.8s), and total step-1-to-summary wall clock dropped 14.5x
    (73.5min -> 5.1min). Recall unchanged (mean recall@k = 0.8619, matching
    entry 11 exactly) -- confirms the fix changes *when* the copy happens,
    not *which* victim gets picked. Unplanned second effect, reported as a
    likely-but-not-independently-proven explanation: pass count itself also
    dropped 12.9x (4,072 -> 316) and `evaluateAdmission`'s own per-call mean
    dropped ~19x despite being untouched by this fix -- read as the slow
    copy having created a backlog-retry feedback loop that inflated both,
    substantially narrowing (not fully closing) entry 11's open
    pass-count-explosion question.
13. [2026-09-01-pass-count-root-cause.md](2026-09-01-pass-count-root-cause.md)
    -- follow-up investigating entry 12's remaining open question directly.
    **Finding 1**: pass count varies 15x (149-2,228) across three
    back-to-back reruns of the identical post-fix binary/config -- entry
    11's "44-250x explosion" framing (one 10M number against a 1M range)
    doesn't hold up once the 10M side is itself shown to vary this much.
    **Finding 2** (new per-pass `examined`/`admitted`/reason logging, added
    tracing-only): during active streaming, the dominant pass-ending reason
    is a *fixed* 256MB per-pass budget cap (~1,300-1,450 candidates
    admitted per capped pass, replacing roughly the whole resident
    population each time) -- since this cap doesn't scale with corpus size
    but total candidate volume does, a 10x bigger corpus mechanically needs
    proportionally more such passes. **Finding 3**: late in the run, most
    passes examine thousands of backlog candidates but admit near-zero
    (hysteresis rejecting a backlog that's gone stale relative to a
    settled resident set) -- expensive-to-scan, cheap-to-resolve passes.
    **Finding 4**: pass count is directly confirmed timing-sensitive, not
    just inferred -- disabling `--quiet-logs` (I/O overhead only, zero
    decision-logic change) alone shifted total passes 2,228 -> 149 in
    back-to-back runs, explaining why entry 12's unrelated per-pass-cost
    fix also moved pass count as a side effect. **Finding 5**: despite that
    15x pass-count swing (149-2,228), total candidates *examined* across a
    whole run stays stable (932K-1.15M, ~1.2x) -- pass count is a
    Coordinator-service-timing chunking artifact, not a proxy for actual
    work; the real cost driver (total examined x per-examination cost) is
    what entry 12 actually fixed, and pass count itself was never a
    reliable health signal. **Conclusion**: the structural drivers (fixed
    budget vs. growing corpus, hysteresis needing to evaluate before it can
    reject) are inherent to any bounded-capacity cache facing a streaming
    workload bigger than it -- not an Arachne-specific defect -- and the
    Coordinator-timing piece only changes chunking, not total work, so
    neither rises to "framework problem"; pass count joins entry 3's own
    5.8x-reproducibility-spread precedent as a noisy-but-harmless metric.
    374/375 suite passing (tracing/logging-only, no regression).
