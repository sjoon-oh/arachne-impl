# Wraparound re-issuance guard for MintAnchorId(), and a correction to the
# previous entry's "~291 years" framing

Follow-up to [2026-08-29-anchor-id-independence.md](2026-08-29-anchor-id-independence.md),
which implemented `Controller::MintAnchorId()` but only guarded against one
of the two collision modes discussed during that design's review: wrapping
back onto payload 0 (the reserved "no Anchor" sentinel). It did not guard
against wrapping onto some *other*, still-meaningful, previously-issued
payload -- the "만일 기존에 발급받은 ID라면 재발급이 필요하다" (if it turns
out to be a previously-issued id, a different one needs to be issued
instead) case raised during that design's review, agreed on at the time but
not actually carried into the code that shipped. This entry closes that gap.

## Correction: the actual safety margin is `2^64` mints, not `2^63`

The previous entry's own report text said the CAS-retry loop protects
against wraparound "across the ... (~291 years at 1 billion mints/second)
`next_anchor_id_` wraparound case." That number assumed the counter's raw
value stays within 63 bits (reserving the top bit the way the *returned* id
does). Re-reading `MintAnchorId()` while implementing this guard found that
assumption wrong: `next_anchor_id_` is a plain, unmasked `std::uint64_t`
counter, and `kAnchorIdBit | payload` is a bitwise OR, not an addition --
once the counter itself climbs past `2^63`, OR-ing the top bit onto it is a
no-op (that bit is already set), so the counter is free to keep counting
all the way to the *actual* 64-bit wraparound at `2^64`, not `2^63`. At 1
billion mints/second that's **~584.9 billion years**, not ~291 -- over 40x
the current age of the universe, not merely "a long time." Both the
`kAnchorIdBit`/`next_anchor_id_`/`MintAnchorId()` doc comments and this
report's own line were corrected to stop citing a specific (and, it turns
out, understated) year figure, since the real point -- "not a realistic
concern for any deployment" -- held under either number anyway.

## What changed

- **`RegionManager::isKnownAnchor(VectorId anchor_id) const`** (new): true
  if `anchor_id` currently depends on at least one Region (`dependencies_`),
  or was released at some point in its lifetime (`anchor_epoch_` -- see that
  member's own doc comment for why entries there are never removed). Takes
  `mutex_` like every other RegionManager bookkeeping call.
- **`Controller::MintAnchorId()`** rewritten around a sticky
  `next_anchor_id_wrapped_` flag (`std::atomic<bool>`, starts `false`):
  the counter increment now also detects (via the same `payload == 0` check
  the previous entry already had, just correctly understood as *the* actual
  wraparound condition rather than an extra guard alongside a separate
  63-bit one) whether this mint just completed a full lap, and latches the
  flag permanently once it does. Every mint before the first (if ever) lap
  returns immediately after the CAS -- `isKnownAnchor()` is never called,
  so `RegionManager::mutex_` is never touched on this path, keeping the
  change's cost exactly zero for every mint that will ever realistically
  happen. Only once the flag is set does a mint additionally check
  `isKnownAnchor()` on the candidate payload, retrying with the next one
  (looping the same CAS-and-check) on an actual collision instead of
  reissuing it.
- Fixed a doc comment on `anchor_epoch_` (`region_manager.hpp`) left over
  from before `MintAnchorId()` existed -- it described "an Anchor id can be
  released and later reused by a new insert()" as the normal case (true
  when `insert()` used to reuse `record.id` directly); reworded to describe
  the current reality (deliberate reuse never happens anymore; the only
  reuse this guards against is the wraparound case above).

## Unit tests

Full suite: **370/371** (same single pre-existing, unrelated failure as
every entry in this investigation). 3 new tests added to
`region_manager_coordinator_test.cpp`, covering the three states an Anchor
id can be in relative to `isKnownAnchor()` -- never assigned, currently
resident, and released-but-permanently-remembered (the exact property
`MintAnchorId()`'s guard depends on):
`IsKnownAnchorIsFalseForAnIdThatWasNeverAssigned`,
`IsKnownAnchorIsTrueWhileAnAnchorCurrentlyDependsOnARegion`,
`IsKnownAnchorStaysTrueAfterReleaseEvenThoughNoLongerResident`.

**Not tested end-to-end**: `MintAnchorId()`'s actual retry-on-collision loop
isn't exercised by driving `next_anchor_id_` to a real wraparound -- doing
so would mean 2^64 actual increments, not practical to run in a unit test.
The tests above cover the primitive (`isKnownAnchor()`) the guard is built
on and its three input states directly; the wraparound-detection arithmetic
itself (`payload == 0` after unsigned overflow) was verified by reading and
reasoning, not by execution, matching how the top-bit reservation itself
was verified in the previous entry.

**New flaky test observed, unrelated to this change**: the first two
full-suite runs immediately after rebuilding (this entry's changes) showed
`StressIndexStage2Test.EvictionCyclingPreservesCorrectnessUnderTinyGpuBudget`
failing, never seen in any earlier round of this investigation. Investigated
before concluding it wasn't a regression: it passed 6/6 when run in
isolation, and 11/11 subsequent full-suite runs (only the first two, right
after the fresh build, failed) -- consistent with a CUDA cold-start artifact
(JIT/driver warm-up) rather than a logic bug, and structurally impossible to
attribute to this entry's own change regardless: `isKnownAnchor()` is
provably never called in any test (none come close to wrapping a 64-bit
counter), so `MintAnchorId()`'s behavior for every test in the suite is
byte-for-byte what it was before this entry, modulo one extra always-false
atomic-bool load per mint. Not fixed (nothing to fix, on this evidence) --
noted here in case it resurfaces and someone goes looking for a cause.

## 1M-scale re-run against raw hnswlib

Same two configs, same flags, same pre-built 1M index as every prior entry
in this investigation, rebuilt against this change (`cpp/build-trace/`).

| | single (batch=1, exec=1, client=1) | multi (batch=32, exec=4, client=4) |
|---|---|---|
| insert ms/op (raw) | 0.2747 | 0.2745 |
| insert ms/op (arachne) | 0.4381 | **0.1786** (beats raw, 1.5x) |
| search ms/op (raw) | 0.1216 | 0.1214 |
| search ms/op (arachne) | **0.0671** (beats raw, 1.8x) | **0.0176** (beats raw, 6.9x) |
| delete ms/op (raw) | 0.0002 | 0.0002 |
| delete ms/op (arachne) | 0.0004 | 0.0005 |
| recall@k (both engines) | matches raw within +0.0001-0.0003 | matches raw within +/-0.0001 |

All figures within normal run-to-run variance of the previous entry's
numbers (insert 0.4471/0.1795, search 0.0623/0.0199 there) -- no regression,
and multi-thread search is fractionally *better* here (6.9x vs. 6.3x).
`RegionManager-releaseAnchor.csv` and `RegionManager-isKnownAnchor.csv` are
both **absent from both trace directories' output** -- the former confirms
the previous entry's delete-path fix still holds; the latter directly
confirms this entry's own fast path held for the entire run (the guard's
extra check genuinely never ran). `Controller-commitRemove`'s max stayed at
essentially zero (0.0001 ms single / 0.0014 ms multi), and
`CostAwareReplacementPolicy-lockwait`'s max stayed at 0.0273 ms in both --
identical to the previous entry, as expected, since this change adds no new
contended path. The previously-flagged `buildRelocationPlan_collect`
final-drain cost (602.9s single / 473.5s multi here) is present at the same
order of magnitude as before, reconfirming that finding as pre-existing and
unrelated to any Anchor-id work.

## Conclusion

The one concrete gap identified against the original design request -- no
actual reissue-collision check on wraparound, only a comment arguing the
scenario was rare -- is closed. The fix is provably zero-cost for every
mint that will ever realistically happen (a plain atomic bool check), and
the 1M-scale re-run confirms no observable change in behavior beyond that.
The design's stated goal (never silently reissue a previously-meaningful
Anchor id) now has an actual mechanism behind it, not just an unrealistic-
odds argument -- for the one gap that mechanism doesn't close (a
currently-*pending*, not-yet-promoted candidate colliding with a fresh mint
on the same wrapped lap), see `isKnownAnchor()`'s own doc comment for why
that residual case was judged not worth plumbing a new query through every
`ReplacementPolicy` implementation.
