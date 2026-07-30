# Arachne C++ Core -- Design Notes

This document explains how the code under `cpp/` is put together, at three
levels of granularity: the **data structures** that carry state, the
**code-level** module layout and how those modules depend on each other, and
the **function-level** behavior of the pieces that matter most.

This is documentation only -- it describes the code as it stands, not a
design proposal.

## 1. Data structure level

### Value types (`include/arachne/types.hpp`)

The lowest layer is a handful of small, trivially-copyable value types
shared by every other module:

- `VectorId`, `RegionId` -- plain `std::uint64_t` aliases. Identity is
  opaque; nothing downstream treats these as anything but keys.
- `VectorView { const float* data; std::uint32_t dim; }` -- a non-owning
  view over one vector's raw components, valid only for the duration of a
  single call (a hand-rolled `std::span<const float>`). Nobody stores a
  `VectorView` past the call that received it.
- `Query { VectorView vector; std::uint32_t top_k; }`
- `Record { VectorId id; VectorView vector; }`
- `Neighbor { VectorId id; float distance; }`
- `SearchResult { std::vector<Neighbor> neighbors; bool served_gpu_only; }`
- `InsertResult { bool ok; }`, `DeleteResult { bool ok; }`

These are the only types that cross every layer boundary (`Engine` ->
`Core` -> `IAdapter`/`IRegion`).

### Region-facing types (`include/arachne/adapter/region.hpp`)

- `ResidencyState` -- `Host | Pending | Resident`: where a Region's
  authoritative mutable state currently lives.
- `RegionFootprint { std::vector<RegionId> regions; }` -- the set of
  regions an operation touched or is scoped to.
- `LeaseHandle { RegionId region; std::uint64_t epoch; bool valid(); }` --
  a GPU write lease. `valid()` is `epoch != 0`; a default-constructed
  handle is always invalid. Opaque outside `Core` and the `IRegion`
  implementation that issued it.
- `ModificationDelta { std::vector<std::byte> payload; }` -- index-specific
  encoding of what changed inside a Region during a lease epoch. Left
  opaque at the `Core` level.
- `ReconciliationReport { bool closed; RegionFootprint touched_neighbors; }`

### Adapter-facing types (`include/arachne/adapter/index_adapter.hpp`)

- `ExecutionMode` -- `GpuOnly | Hybrid`.
- `TraverseRequest { Query query; ExecutionMode mode; RegionFootprint scope; }`
- `TraverseResult { SearchResult result; RegionFootprint touched; bool completed_within_scope; }`
- `ModifyOp` -- `Insert | Delete`.
- `ModifyRequest { ModifyOp op; Record record; VectorId target; ExecutionMode mode; RegionFootprint scope; LeaseHandle lease; }`
- `ModifyResult { bool ok; RegionFootprint touched; RegionFootprint modified; }`

### Core-owned types (`include/arachne/core/anchor_manager.hpp`)

- `Stitch { RegionId region; LeaseHandle lease; }` -- the association
  between an Anchor (a `VectorId`) and a Region it currently holds GPU
  write authority over. `AnchorManager` stores these keyed by anchor id:
  `std::unordered_map<VectorId, std::vector<Stitch>>`, guarded by a plain
  `std::mutex`.

### RoutingCache's internal data structure (`src/core/routing_cache_hnsw.cpp`)

`ASRoutingCacheHnsw::Instance` is the private, `.cpp`-only wrapper around one
hnswlib `HierarchicalNSW<float>` index:

```
Instance
  hnswlib::L2Space space_
  hnswlib::HierarchicalNSW<float> index_
  std::unordered_set<VectorId> live_ids_
  std::size_t tombstones_
```

`ASRoutingCacheHnsw` itself holds exactly one such `Instance` behind a
`std::unique_ptr<Instance> active_`, protected by a `std::shared_mutex
mutex_`, plus background-compaction bookkeeping: `std::atomic<bool>
compacting_` and a `std::thread compaction_thread_`. There is deliberately
no second "shadow" member on `ASRoutingCacheHnsw` -- the shadow `Instance` is
a purely local variable inside `compactImpl()` and only ever becomes
`active_` at the moment of swap.

## 2. Code level -- module layout and dependencies

```
include/arachne/
  types.hpp                        (no deps)
  logging.hpp                      (no deps; branches on ARACHNE_WITH_RAFT)
  adapter/
    region.hpp                     -> types.hpp
    index_adapter.hpp               -> region.hpp, types.hpp
  core/
    routing_cache.hpp               -> types.hpp
    routing_cache_hnsw.hpp          -> routing_cache.hpp
    anchor_manager.hpp               -> adapter/region.hpp, types.hpp
    core.hpp                         -> adapter/index_adapter.hpp, core/anchor_manager.hpp, core/routing_cache.hpp
  interface/
    index.hpp                         (no deps besides types.hpp)
    engine.hpp                       -> adapter/index_adapter.hpp, core/core.hpp, core/routing_cache.hpp, interface/index.hpp

src/
  core/routing_cache_hnsw.cpp        -> routing_cache_hnsw.hpp, <hnswlib/hnswlib.h>
  core/anchor_manager.cpp            -> anchor_manager.hpp
  core/core.cpp                      -> core.hpp, logging.hpp
  interface/engine.cpp                -> engine.hpp
```

Three architectural layers, one direction of dependency (top depends on
bottom, never the reverse):

1. **Interface** -- `Index`, `Engine`. `Index` is the abstract entry point
   application code programs against -- pure-virtual `search`/`insert`/
   `remove`, nothing else -- mirroring the same pattern `RoutingCache`
   uses one layer down: it exists so application code stays
   implementation-agnostic about which concrete top-level implementation
   actually serves requests, the same way `Core` stays agnostic about
   which concrete `RoutingCache` it's handed. `Engine` is the first (and
   currently only) `Index` implementation. It owns an `IAdapter` and a
   `RoutingCache` (both injected as `std::unique_ptr`, so the caller picks
   the concrete implementations), and owns a `Core` by value that is
   constructed from references to those two. `Engine`'s three overrides
   (`search`/`insert`/`remove`) do nothing but forward to `controller_`.

2. **Core** -- `Core`, `AnchorManager`. `Core` is the index-agnostic
   control plane: it decides *where* (Host/GPU/hybrid, which regions)
   SEARCH/INSERT/DELETE run. It is written only against the
   `IAdapter`/`IRegion` interfaces and the `RoutingCache` interface --
   never against a concrete index or a concrete cache. It holds:
   - `IAdapter& adapter_` -- injected, not owned.
   - `RoutingCache& routing_cache_` -- injected, not owned. Pluggable:
     answers only "is this query close to something we've seen."
   - `AnchorManager anchor_manager_` -- owned by value, not pluggable.
     Core's own policy state: which Stitches (write leases) each Anchor
     currently holds.
   - `next_anchor_id_`, `drift_window_host_`, `drift_window_total_` --
     small bits of Core's own bookkeeping (id allocation, a rolling
     Host-vs-GPU traversal counter for future drift-triggered promotion).

   `AnchorManager` is a small, self-contained, mutex-guarded map. It has
   no knowledge of `RoutingCache`, `IAdapter`, or anything above it --
   it only understands `VectorId -> Stitch` bookkeeping.

3. **Adapter** -- `IAdapter`, `IRegion` (interfaces only; no concrete
   implementation exists yet in this tree -- integrating a real index is
   left as separate future work). `ASRoutingCacheHnsw` also lives
   conceptually at this level (it is a concrete *implementation* plugged
   into `Core` via the `RoutingCache` interface), backed by the vendored
   `thirdparty/hnswlib` submodule (pinned to release `v0.9.0`). hnswlib is
   a `.cpp`-only dependency: `routing_cache_hnsw.hpp` forward-declares the
   nested `Instance` type and never includes `<hnswlib/hnswlib.h>`, so
   nothing that only includes the header needs hnswlib on its include
   path.

### Build (`CMakeLists.txt`, `conda/arachne-blackwell.yml`)

- `project(arachne LANGUAGES CXX CUDA)`, C++20/CUDA20, default
  `CMAKE_CUDA_ARCHITECTURES` set to Blackwell (`100 120`), overridable.
  Dependencies (`raft`, `spdlog`, `GTest`) resolve from the active conda
  environment via `CMAKE_PREFIX_PATH`.
- `hnswlib` is an `INTERFACE` target (header-only, vendored, not on conda)
  linked `PRIVATE` into `arachne_core`, so it never leaks onto consumers'
  include paths.
- `arachne_core` is the one library, built from `interface/engine.cpp`,
  `core/core.cpp`, `core/anchor_manager.cpp`, `core/routing_cache_hnsw.cpp`.
  Aliased as `arachne::core`.
- `ARACHNE_USE_RAFT` (default `ON`) switches the logging backend and links
  `raft::raft`; when off, falls back to standalone `spdlog`.
- `ARACHNE_BUILD_TESTS` (default `ON`) adds `test/`, which builds
  `arachne_tests` against GoogleTest (`GTest::gtest`, `GTest::gtest_main`,
  discovered via `gtest_discover_tests`).
- `Threads::Threads` is linked publicly for `ASRoutingCacheHnsw`'s
  `std::thread`/`std::shared_mutex` background compaction.

### Formatting convention

All hand-written files under `include/` and `src/` (excluding the vendored
`thirdparty/hnswlib` submodule) use **tab** indentation, including
`CMakeLists.txt` files.

## 3. Function level -- the pieces that matter most

### `Index` (`interface/index.hpp`)

```cpp
virtual SearchResult search(const Query& query) = 0;
virtual InsertResult insert(const Record& record) = 0;
virtual DeleteResult remove(VectorId id) = 0;
```

Pure interface, no state, no `.cpp` file. Just a virtual destructor and the
three request methods. Deliberately as thin as `RoutingCache`: it exists
only to let callers depend on "something that can search/insert/remove"
rather than on `Engine` specifically.

### `Engine` (`interface/engine.hpp` / `.cpp`)

```cpp
Engine(std::unique_ptr<IAdapter> adapter, std::unique_ptr<RoutingCache> routing_cache);
SearchResult search(const Query& query) override;
InsertResult insert(const Record& record) override;
DeleteResult remove(VectorId id) override;
```

Takes ownership of both injected dependencies, constructs `controller_` from
references to them (`controller_(*adapter_, *routing_cache_)`), and every public
method is a one-line forward to the corresponding `controller_` method. No logic
of its own.

### `RoutingCache` (`core/routing_cache.hpp`)

The abstract interface answering exactly one question -- "is the incoming
query vector close enough to a previously seen one (an Anchor, identified
only by its `VectorId`) that routing to GPU is worthwhile?" It does not
know what an Anchor id means beyond that: no Stitch/write-lease
bookkeeping, no eviction policy.

```cpp
virtual std::optional<VectorId> nearest(const VectorView& query) = 0;
virtual VectorId ensure(VectorId id, const VectorView& vector) = 0;
virtual void erase(VectorId id) = 0;
```

- `nearest` -- id of the closest registered entry, or `nullopt` if none is
  close enough (or the cache is empty). "Close enough" is entirely an
  implementation decision; `Core` applies no threshold of its own.
- `ensure` -- idempotent get-or-create: returns the existing id if a close
  enough entry already exists, otherwise registers `vector` under `id`
  and returns `id`.
- `erase` -- removes an entry; a no-op if `id` is unknown.

### `ASRoutingCacheHnsw` (`core/routing_cache_hnsw.hpp` / `.cpp`)

The concrete `RoutingCache` implementation, backed by hnswlib, with three
responsibilities layered on top of the raw index:

**Concurrency.** hnswlib's own internal locking protects concurrent reads
against each other and concurrent writes against each other, but *not*
concurrent reads against concurrent writes (`searchKnn` walks the link
list without taking any of hnswlib's locks; confirmed by reading
`hnswalg.h`). `ASRoutingCacheHnsw` compensates with its own
`std::shared_mutex mutex_`: `nearest()` takes a `shared_lock`,
`ensure()`/`erase()` take a `unique_lock`. hnswlib is never touched outside
one of those two lock modes.

- `nearest(query)`: shared lock, delegates to `active_->findNearest`.
- `ensure(id, vector)`: unique lock; if `findNearest` already returns a
  match, returns that id (the requested `id` is discarded -- the caller's
  proposed id only wins when no existing entry is close enough).
  Otherwise inserts under `id` and returns `id`.
- `erase(id)`: unique lock; calls `active_->erase(id)`, which
  `markDelete`s in hnswlib and increments a tombstone counter. After
  releasing the lock, if the tombstone/(tombstone+live) ratio has crossed
  `max_tombstone_ratio_`, calls `triggerCompaction()`.

**Deletion / compaction.** hnswlib only tombstones (`markDelete`) -- no
real space reclamation happens in the underlying library. Once the
tombstone ratio crosses the configured threshold, `ASRoutingCacheHnsw`
rebuilds a fresh "shadow" `Instance` from the active one's live entries on
a background thread and swaps it in. `compactImpl()` runs in three phases:

1. **Snapshot** -- brief `shared_lock`; walks `active_->forEachLive` to
   copy every surviving `(id, vector)` pair. Readers and the next writer
   are only blocked for this copy, not for the rebuild.
2. **Rebuild** -- *no lock held at all*. Constructs a new `Instance` with
   capacity `max(snapshot.size() * 2, initial_capacity_)` and inserts
   every snapshotted pair into it. This is the expensive part, and both
   readers and writers of `active_` proceed fully concurrently while it
   runs.
3. **Reconcile and swap** -- brief `unique_lock`. Diffs `active_`'s
   current live-id set against what was migrated: anything erased from
   `active_` during the rebuild is erased from the shadow too; anything
   inserted into `active_` during the rebuild is copied into the shadow.
   Then `active_ = std::move(shadow)`. This phase's cost is bounded by how
   much changed during the rebuild, not by the index size.

Compaction never has to preserve any per-id state beyond `(id, vector)`,
since Stitch bookkeeping lives in `AnchorManager`, not here.

`triggerCompaction()` uses `compacting_.compare_exchange_strong(expected=false, true)`
so at most one compaction thread runs at a time; a duplicate trigger while
one is already running is silently dropped. `waitForCompaction()` (not
part of the `RoutingCache` interface) joins the background thread; used by
tests and as a graceful-shutdown synchronization point. The destructor
also joins if a thread is still joinable.

### `AnchorManager` (`core/anchor_manager.hpp` / `.cpp`)

Owns Stitch bookkeeping for whatever Anchor ids `Core` hands it. All four
methods take a plain `std::lock_guard<std::mutex>` (no shared/exclusive
distinction needed -- there's no expensive background operation to allow
concurrent readers through, unlike `ASRoutingCacheHnsw`).

```cpp
std::vector<Stitch> stitchesOf(VectorId anchor_id) const;
void addStitch(VectorId anchor_id, RegionId region, LeaseHandle lease);
LeaseHandle removeStitch(VectorId anchor_id, RegionId region);
std::vector<Stitch> forget(VectorId anchor_id);
```

- `stitchesOf` -- returns a copy of the anchor's current Stitch list (empty
  if none).
- `addStitch` -- idempotent per region: if `anchor_id` already has a
  Stitch on `region`, this is a no-op (the existing lease is kept, not
  overwritten).
- `removeStitch` -- removes the Stitch for one region, returning its
  `LeaseHandle` so the caller can release the underlying `IRegion` lease.
  Returns a default (invalid) handle if there was nothing to remove. If
  the anchor's Stitch list becomes empty, the map entry itself is erased.
- `forget` -- removes and returns *every* Stitch for an anchor in one
  call; the hook a future eviction policy would use to reclaim a cold
  Anchor's leases wholesale, rather than calling `removeStitch` in a loop.

### `Core` (`core/core.hpp` / `.cpp`)

The control plane. Six private methods map directly onto the four Quick
Summary design points; `search`/`insert`/`remove` are the public surface.

```cpp
SearchResult search(const Query& query);
InsertResult insert(const Record& record);
DeleteResult remove(VectorId id);
```

- **`route(query) -> RoutingDecision`** (design point 1). Calls
  `routing_cache_.nearest(query.vector)`; if it returns an anchor id,
  looks up that anchor's Stitches via `anchor_manager_.stitchesOf`. If
  there are any, marks the query GPU-only-eligible and collects every
  Stitch's region into `predicted_scope`.
- **`search(query)`**. Calls `route()`; if eligible, issues a
  `TraverseRequest` scoped to `predicted_scope` in `GpuOnly` mode.
  If that traversal reports `!completed_within_scope`, falls back to a
  second, `Hybrid`-mode traversal so the caller still gets a complete
  answer. Records the traversal outcome via `recordTraversalForDrift`,
  then registers the query vector as a (possibly new) Anchor via
  `routing_cache_.ensure(next_anchor_id_++, query.vector)` -- this happens
  *after* the search, so a query is only ever routed off Anchors that
  existed before it arrived.
- **`insert(record)`**. Ensures an Anchor for the record's vector first
  (`routing_cache_.ensure`). If that anchor already holds a *valid*
  Stitch, scopes the modification request to that single region in
  `GpuOnly` mode and attaches the lease; otherwise the request stays
  `Hybrid` with no scope. (Only the first valid Stitch found is used --
  multi-region inserts are explicitly left as future work.) After a
  successful `adapter_.modify()`, calls `make(anchor_id, region)` for
  every region the adapter reports as actually modified.
- **`remove(id)`**. Issues a `Delete`-mode `ModifyRequest` directly;
  no Anchor/routing involvement (deletion targets an id, not a vector to
  route by proximity).
- **`recordTraversalForDrift(touched_host)`** (design point 2 trigger).
  Maintains a rolling window (`kDriftWindowSize = 128`) counting how many
  of the last N traversals touched Host vs. total; resets when the window
  fills. Not yet consumed by any promotion/eviction trigger -- the
  counters exist but nothing reads them yet.
- **`promote(footprint)` / `evict(footprint)`** (design point 2, the
  "how", not the "when"). Placeholder policies: `promote` calls
  `materializeOnDevice()` on every region in the given footprint;
  `evict` calls `evictFromDevice()`. Not yet wired to any trigger.
- **`verify(query, anchor_id, gpu_only_result)`** (design point 3). Not
  yet called from `search()`. Re-runs the query in `Hybrid` mode as ground
  truth and compares neighbor-id sequences. On any mismatch, calls
  `anchor_manager_.forget(anchor_id)` to reclaim every Stitch on that
  anchor (its regions no longer represent the anchor's true locality), and
  releases each reclaimed lease via `IRegion::releaseWriteLease`.
- **`make(anchor_id, region) -> bool`** (design point 4). First checks
  `anchor_manager_.stitchesOf(anchor_id)` for an existing Stitch on
  `region` -- if found, returns `true` immediately without touching the
  adapter (this ordering matters: checking *before* acquiring a lease
  avoids leaking one if the region turns out to already be stitched).
  Otherwise resolves the region, requires it to be `Resident`, acquires a
  write lease, and on success records the Stitch via
  `anchor_manager_.addStitch`. Returns `false` if the region isn't
  resolvable, isn't Resident, or lease acquisition fails.

**Known, documented limitation**: `make()`'s check-then-acquire sequence
has a TOCTOU race window if two threads try to stitch the same anchor to
the same region concurrently -- both could pass the "not yet stitched"
check before either calls `addStitch`. The result would be a harmless
duplicate `acquireWriteLease()` call, not a crash; left unfixed as
low-severity, since a real fix requires deciding where the necessary
per-(anchor, region) lock should live.

### Logging (`logging.hpp`)

`ARACHNE_LOG_TRACE/DEBUG/INFO/WARN/ERROR(...)` macros, fmt-style `{}`
placeholders either way:

- `ARACHNE_WITH_RAFT` defined (RAFT linked): forwards to `RAFT_LOG_*`,
  which drives `raft::default_logger()` -- a rapids-logger/spdlog wrapper
  -- so Arachne's control-plane logs and RAFT's GPU-primitive logs share
  one sink/pattern/level.
- Otherwise: forwards to `SPDLOG_LOGGER_*` against a standalone,
  lazily-created `arachne::default_logger()` (one process-wide
  `spdlog::stderr_color_mt("arachne")`).

## Testing

`test/routing_cache_hnsw_test.cpp` (10 cases) and
`test/anchor_manager_test.cpp` (9 cases), both GoogleTest, both built into
one `arachne_tests` binary via `test/CMakeLists.txt`.

Notably: `ASRoutingCacheHnswTest.CompactionKeepsSurvivingIdsQueryableAndDropsErasedOnes`
drives a real compaction (via a low `max_tombstone_ratio`) and asserts
surviving ids remain queryable and erased ids don't resurrect after the
active/shadow swap; `ConcurrentEnsureNearestEraseDoesNotRace` runs 8
threads x 200 mixed `ensure`/`nearest`/`erase` operations and is meant to
be run under ThreadSanitizer to catch the exact read/write hazard
`ASRoutingCacheHnsw`'s `shared_mutex` exists to prevent.
