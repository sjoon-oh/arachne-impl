# 10M-scale pre-built indices (for future experiments)

Status: **in progress**. Not an investigation entry on its own -- this
records what was pre-built, with what parameters, and where, so a later
session can `--load-index` straight into a real experiment without repeating
multi-hour graph construction. Requested alongside the
`shouldYieldPass()` fix (see
[2026-08-28-latency-tracing.md](2026-08-28-latency-tracing.md)) but is an
independent, unrelated piece of prep work -- these graphs are not used by
that entry's verification runs.

Exported files live alongside this entry, under
`2026-08-29-10m-index-builds/` (same base name convention as every other
report entry with attached artifacts).

## New tool: `arachne_build_and_export_index`

`cpp/test/index/build_and_export_index.cpp` (+ matching `CMakeLists.txt`
target `arachne_build_and_export_index`) -- a minimal build-only-and-export
binary, written for this because `hnsw_workload_compare` always replays the
full streaming workload after building/loading and has no "build then stop"
mode. Does exactly one thing: read a workload's base pool, `HnswlibIndexGpu::
build()` it, `exportTo()` the result, exit. No Controller, no GPU residency,
no streaming -- confirmed elsewhere in this investigation that `build()` is
entirely independent of all of that.

**Capacity gotcha (documented in the tool's own `--capacity` flag comment,
re-noted here since it bit the smoke test below):** hnswlib's
`max_elements` is fixed at construction and `loadFrom()` requires an exact
match against whatever capacity the *loading* adapter was constructed with
-- not just "big enough". A graph exported with one capacity cannot be
loaded by an adapter constructed with a different one (throws
`std::invalid_argument`). Default here is `limit_base * 1.3` (matches this
investigation's own 1M-scale build: 1,000,000 base : 1,300,000 capacity, for
30 steps of 10,000 inserts each -- see `sweep_cost_aware_1m.sh`). **Whoever
loads these 10M exports later must construct their `HnswlibIndexGpu`/pass
`--capacity` to `hnsw_workload_compare` (once it grows a matching override,
or by picking `--limit-steps` such that `ComputeSizing()` lands on the same
number) with exactly 13,000,000** -- the same value these builds used
(`10,000,000 * 1.3`, `--capacity` left at its default, not overridden).

Smoke-tested at 5000-base scale (build -> exportTo -> a real
`hnsw_workload_compare --load-index` round trip) before starting the real
10M builds -- confirmed working once the capacity value matched on both
sides (the mismatch above is expected/by-design, not a bug).

## The three variants (all against the *full* 10,000,000-row SIFT1B base
pool, `align_workload_a_10m/set_1`)

| variant | M | ef_construction | output file | capacity | build time | file size | status |
|---|---|---|---|---|---|---|---|
| A | 32 | 128 | `arachne_10m_index_m32_efc128.bin` | 13,000,000 | 61.37 min | 4.0 GB | **done** |
| B | 32 | 64  | `arachne_10m_index_m32_efc64.bin`  | 13,000,000 | 34.41 min | 4.0 GB | **done** |
| C | 16 | 64  | `arachne_10m_index_m16_efc64.bin`  | 13,000,000 | 29.08 min | 2.8 GB | **done** |

All three launched in parallel (independent background processes, each
effectively single-threaded CPU-bound work -- `HnswlibIndex::build()`'s own
current implementation doesn't parallelize `addPoint()` calls across
threads, matching every other build in this investigation) rather than
sequentially, to reduce total wall-clock time; the machine has 16 cores so
3 concurrent single-threaded builds left headroom (confirmed no meaningful
slowdown from running together -- variant C's 29.08 min here is in the same
range as this investigation's earlier solo 1M-scale build extrapolated by
~10x, not inflated by contention). All three completed and exported without
error. **Not yet round-trip-verified at this 10M scale specifically** -- the
`--load-index` mechanism itself was smoke-tested earlier (5000-base scale,
see above), which is what these 3 builds reuse unchanged, but no
`hnsw_workload_compare --load-index` call has actually been made against any
of these 3 files yet. Whoever uses one of these for a real experiment should
expect it to work (same code path, same capacity convention) but hasn't had
that specific confirmation.

**10.8 GB combined, not added to git** -- same reasoning as the 1M index
backup in the latency-tracing entry (too large for a git-tracked report
directory). Left on disk under this entry's own directory as requested;
delete or relocate at your discretion once no longer needed for a future
10M-scale experiment.

M=32 costs roughly 1.7-2.1x M=16's build time at the same ef_construction
(29.08 min at M=16/efc=64 vs 34.41 min at M=32/efc=64 is *not* quite that
ratio -- ef_construction dominates more than M does at this data scale;
efc=128 vs 64 at the same M=32 very nearly doubled build time, 34.41 -> 61.37
min, close to linear in ef_construction as expected).
