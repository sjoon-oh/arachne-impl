# Arachne

**Arachne is a GPU-accelerated control plane for streaming Approximate
Nearest-Neighbor Search (ANNS).** It does not implement a new ANNS
algorithm. Instead, it sits in front of an existing index (currently
[hnswlib](https://github.com/nmslib/hnswlib)) and decides, query by query,
*where* traversal and modification should run (CPU or GPU) and *which* part
of the index should be resident on the GPU at any given moment.

> This is an active research implementation, not a finished system. See
> [Status](#status) for what is and isn't wired up yet.

## Table of contents

- [Motivation](#motivation)
- [Core idea](#core-idea)
- [Key design points](#key-design-points)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Building](#building)
- [Status](#status)
- [License](#license)

## Motivation

A large ANNS index does not fit in GPU memory. Systems that combine a CPU
index with a GPU accelerator typically fall back to the host whenever the
data a query needs isn't already on the device, paying for repeated
host↔GPU transfers and synchronization along the way.

Streaming vector workloads have a property that hybrid CPU/GPU systems
generally don't exploit: **queries arrive with semantic locality**.
Similar queries tend to touch similar regions of the index in succession.
If Arachne keeps the index state a query actually used resident on the
GPU, a subsequent, sufficiently similar query can often be answered
entirely on the GPU — no host round trip needed. The same locality applies
to insertion: if both the traversal and the modification footprint of an
insert stay inside GPU-resident territory, the GPU can perform the
insertion itself.

The catch is that locality drifts. As the query distribution shifts, data
that used to be hot on the GPU stops being useful, and repeated GPU-only
hits stop being trustworthy proof of correctness, because GPU-only
execution never re-checks the host's authoritative index. Arachne's job is
to exploit locality *and* continuously validate it, re-adjusting GPU
residency as the workload moves.

## Core idea

Arachne decomposes every ANNS operation into two primitives that any
underlying index already exposes in some form:

```
Index Traversal   -- explore index state related to a query or insertion target
Index Modification -- change index state for an insertion or deletion

SEARCH = Traversal
INSERT = Traversal -> Modification
DELETE = Modification
```

The traversal and modification *algorithms* remain entirely the
responsibility of the underlying index (`IAdapter` in this codebase).
Arachne's own responsibility is the **control plane** around those
primitives: given the locality and execution footprint an index adapter
exposes, decide whether an operation can be completed on the GPU alone,
which index state should be promoted to or evicted from the GPU, and when
GPU-only execution needs to be revalidated against the host.

## Key design points

**1. Anchor-Query-based semantic routing cache.** Every incoming query
first passes through a lightweight routing cache — itself a small
embedded ANNS structure — that stores past *Anchor Queries* and the index
footprint each one touched. If a new query is close enough to an existing
Anchor and that Anchor's footprint is currently GPU-resident, the query is
routed GPU-only. Otherwise it falls through to the underlying index's
normal hybrid traversal, and whatever footprint that traversal actually
touches becomes new locality evidence.

**2. Drift-driven promotion and eviction.** When the fraction of work
falling back to the host rises, Arachne treats that as evidence of
workload drift and re-balances GPU residency. Promotion moves only the
minimal subset of host-side state needed to cover GPU-only execution for
more Anchors — not an Anchor's entire footprint. Eviction targets whatever
state, if removed, would shrink total Anchor coverage the least — not
whatever was merely least-recently or least-frequently used in isolation.

**3. Selective quality verification.** A long streak of GPU-only hits is
not by itself proof of quality: GPU-only execution can only see what's
already resident, so it cannot tell whether the host index holds a better
candidate. Arachne periodically routes a query through the verification
path — a real host or stronger hybrid traversal — and compares results
against the GPU-only answer. Verification cadence can react to signals
such as time since last check, consecutive GPU-only hits, update frequency
of the relevant region, Anchor confidence, and the GPU-only result's
distance/rank margin.

**4. GPU write leases for local modification.** If an insertion's
candidate traversal is GPU-resolvable and its actual modification target
stays inside GPU-resident territory, Arachne grants that region a
temporary GPU write lease and lets the GPU perform the insertion directly,
accumulating changes for the lease's lifetime before writing back to the
host on eviction, dirty-state pressure, lease revocation, or a maintenance
trigger. Connectivity/invariants that reach outside GPU-owned territory
are repaired by a background reconciliation step; if reconciliation isn't
cheap or the modification isn't sufficiently self-contained, Arachne falls
back to the ordinary CPU-authoritative update path.

Taken together, Arachne is not a new index — it's a **vector-streaming
control plane** that uses query-conditioned semantic locality to decide
GPU residency, traversal placement, and temporary write authority on top
of whatever index an adapter wraps.

## Architecture

```
Application
   |
   v
Index (interface/index.hpp)        -- abstract entry point (search/insert/remove)
   |
   v
Controller (core/controller.hpp)   -- dispatches ops, owns the promote/evict loop
   |            \
   v             v
RoutingCache   RegionManager -- Anchors, dependency graph, promotion/eviction,
(Anchor query    |               compaction, backed by a pluggable ReplacementPolicy
 pre-filter)     v               (FIFO / LRU / LFU / Clock / 2Q)
              OpScheduler -- batches/dispatches Traverse & Modify work
                 |
                 v
              IAdapter -- wraps the underlying index (e.g. hnswlib);
                          owns the real traversal/modification algorithms
                 |
                 v
              DeviceRegionPool / DeviceContext (gpu/) -- GPU memory residency,
                          built on RAFT + RMM
```

- **`interface/`** — `Index`, the abstract entry point application code
  targets; `IndexImpl`/`Engine` is the concrete implementation built on
  `core/`.
- **`core/`** — the control plane itself:
  - `Controller` dispatches `search`/`insert`/`remove`, drives
    `RoutingCache` lookups, and coordinates promotion/eviction.
  - `RegionManager` owns Anchors and their Region dependency graph, and is
    the single point of truth actual promotion/eviction/compaction
    decisions are validated against — a `ReplacementPolicy` only ever
    *suggests* candidates.
  - `ReplacementPolicy` is a pluggable eviction/promotion strategy
    (`FifoReplacementPolicy`, `LruReplacementPolicy`,
    `LfuReplacementPolicy`, `ClockReplacementPolicy`,
    `TwoQReplacementPolicy`), each written to minimize lock contention on
    its own hot path.
  - `RoutingCache` (`ASRoutingCache`, backed by `ASRoutingCacheHnsw`) is
    the Anchor-Query semantic pre-filter described above.
  - `OpScheduler` batches and dispatches traversal/modification work
    across worker threads.
- **`adapter/`** — `IAdapter`/`IRegion`, the interface an index
  implementation (e.g. an hnswlib wrapper) implements to plug into
  Arachne.
- **`gpu/`** — `DeviceContext` (RAFT device resources + RMM memory
  resources) and `DeviceRegionPool` (handle-based GPU memory residency:
  allocate/acquire/free/compact for the Regions Arachne is managing).
- **`telemetry/`** — optional (compile-time gated, see
  [Building](#building)) fine-grained latency and lock-contention tracing,
  emitting one `<module>-<feature>.csv` file per traced call site,
  intended for building the breakdown graphs a systems paper needs rather
  than for production use.

## Repository layout

```
cpp/
  include/            Public headers, mirrors src/ below
  src/
    interface/         Index / IndexImpl
    core/               Controller, RegionManager, ReplacementPolicy,
                         RoutingCache, OpScheduler
    gpu/                DeviceContext, DeviceRegionPool
    telemetry/          Opt-in CSV tracing (ARACHNE_ENABLE_TRACING)
    util/               CPU-side SIMD distance kernels (Highway)
  test/                gtest suite
  thirdparty/          Vendored dependencies (hnswlib, ...)
conda/
  environment.yml      Conda environment for the CUDA/C++ build
discussion/            Design notes / paper draft (background reading, not
                       required to understand this README)
```

## Building

Arachne targets NVIDIA Blackwell GPUs (SM100 data-center, SM120 consumer)
and requires CUDA 12.8+.

```bash
conda env create -f conda/environment.yml
conda activate arachne
cmake -S cpp -B cpp/build -GNinja
cmake --build cpp/build
ctest --test-dir cpp/build
```

Notable CMake options (see `cpp/CMakeLists.txt`):

- `ARACHNE_BUILD_TESTS` (default `ON`) — build the gtest suite.
- `ARACHNE_ENABLE_TRACING` (default `OFF`) — compile in the fine-grained
  latency/lock-contention CSV tracing described under
  [Architecture](#architecture). Off by default so a normal build carries
  zero tracing overhead; every `ARACHNE_TRACE_SCOPE()` call site compiles
  to nothing at all when disabled.

## Status

This is a research/systems-paper implementation under active development,
not a production-ready library. The core control-plane skeleton
(`Controller` / `RegionManager` / `ReplacementPolicy` / `DeviceContext` /
`DeviceRegionPool`) is in place and covered by an extensive gtest suite,
but several pieces described in [Key design points](#key-design-points)
are not fully wired end-to-end yet — most notably:

- `IAdapter` implementations do not yet have a path to actually touch GPU
  memory (the plumbing from `Controller`/`DeviceRegionPool` down to a
  concrete adapter such as an hnswlib wrapper).
- How an underlying index's graph should be partitioned into `Region`s
  (locality-aware clustering vs. arbitrary id ranges) is still an open
  design question.
- Statistics/telemetry that a smarter `ReplacementPolicy` would consume
  (per-Anchor hotness, GPU-only hit/miss history, verification outcomes)
  are only partially collected.

See `cpp/todo.txt` for the current, more detailed working list.

## License

MIT — see [LICENSE](LICENSE).
