# `arachne` (Python) — Submodule Overview and Usage

`python/arachne` is made up of three submodules.

```text
python/arachne/
  workload/     offline preprocessing tool that turns a dataset into an experimental
                workload (base/insert/query pools, streaming insert(+delete)+search)
                (implemented, the focus of this doc)
    example/      runnable sample scripts (see "Quickstart" below)
  benchmark/    orchestration scripts that take the pools workload produces and actually
                run cpp/example binaries, collecting logs
                (still a placeholder, not implemented)
  analysis/     post-processing tool that parses the logs benchmark leaves behind and
                aggregates/plots miss rate / latency / throughput etc.
                (still a placeholder, not implemented)
```

The relationship between them mirrors the pipeline order: **workload (data prep) → benchmark (execution) → analysis (result analysis)**.
This document covers how to use `workload`, which is the one that's implemented.

---

## Quickstart

A runnable example lives in `workload/example/`. With no arguments it runs the entire insert(+delete)+search streaming pipeline end to end against a small synthetic dataset the code generates itself — no real dataset needed. It runs both `WorkloadKind.INSERT_SEARCH` and `WorkloadKind.INSERT_DELETE_SEARCH`, each producing one groundtruth checkpoint per step (see "Streaming workloads" below):

```bash
cd arachne   # top level of this repository (the arachne project)
PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py
```

Running it produces a synthetic base dataset, a separate synthetic evaluation query file, and one workload generation (`set_1`) per workload kind under `python/arachne/workload/example/output/` (this directory is gitignored, so it's safe to delete and re-run anytime).

To point the script at real dataset files instead of the synthetic demo, pass `--config` with an `.ini` file instead of editing the script (see "Config files (`.ini`)" below). `workload/example/quickstart_streaming_workload.ini` is a simple, immediately-runnable example (it points at the same synthetic files the demo above just wrote) — `workload/example/streaming_workload.example.ini` is a fully annotated template meant to be copied and pointed at a real dataset:

```bash
PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py \
    --config python/arachne/workload/example/quickstart_streaming_workload.ini
```

If the input file formats (xbin/vecs/SPACEV) are confusing, also try `example/inspect_source_formats.py` — a small script that hand-writes an fvecs file and a SPACEV-style file and verifies they read back exactly right:

```bash
PYTHONPATH=python python3 python/arachne/workload/example/inspect_source_formats.py
```

The sections below walk through what these example scripts do internally, step by step.

---

## What `arachne.workload` does

In one sentence: **a tool that takes "one raw dataset" as input and produces, as output files, "base/insert/query pools cut to exactly the sizes an experiment needs, plus the ground-truth answers for a separate evaluation query set."** It never touches any downstream C++/CUDA ANN library source code — the goal is to hand it output in a file format it can already read.

### Why this is needed (background)

A streaming ANN benchmark needs a few distinct kinds of vector sets:

1. **Base pool** — the vectors used to build the initial index.
2. **Insert pool** — vectors to be streamed in one at a time / in batches after the build.
3. **Stream queries** — held-out vectors available to submit as SEARCH traffic during a benchmark run (e.g. to measure latency/throughput under concurrent insert/delete). **Must never be inserted** (otherwise you'd be "searching for something already in the index," which contaminates the results).
4. **Evaluation queries** — a separate, already-final query set used purely to grade recall/ground truth (see "Two kinds of queries" below).

Public datasets (SIFT1B, etc.) either don't already give you base/insert pools split out at the scale one experiment needs (e.g. a 1B-row base file, when an experiment only wants 1M), or don't let you control locality (which cluster something was drawn from). `arachne.workload`'s job is to cut the base/insert/stream-query pools directly **out of one raw file, in exactly the counts wanted, under exactly the locality conditions wanted** (the raw file's binary format itself is covered in the next section).

### Two kinds of queries: stream queries vs. evaluation queries

It's tempting to carve "the query set" out of the same base dataset file as everything else (held out, never inserted). `arachne.workload` deliberately does **not** use that held-out slice to grade ground truth. Instead:

- **`query_pool`** — still cut from the base dataset via the same clustering split as base/insert, held out and never inserted. This is **stream traffic**: vectors available to submit as SEARCH requests interleaved with INSERT/DELETE during an actual benchmark run.
- **`eval_query_pool`** — read from a **separate, already-final query file** (`run()`'s `eval_query_dataset_path`), exactly mirroring how real datasets ship base and query vectors as two distinct files (e.g. a 1B-row base file plus its own 10K-row query file). This is what ground truth is computed against, at every checkpoint.

Keeping these separate means the query set used for grading is never entangled with whatever locality/clustering decisions were made for the base/insert split, and matches the natural shape of real ANN datasets, which never ask you to carve a query set out of their base file yourself.

### Input file format: what "xbin" actually is (and vecs/SPACEV support)

`arachne.workload` doesn't invent any format of its own. Both source formats it currently supports (`SourceFormat`) are pre-existing layouts already widely used across the ANN benchmark ecosystem. The same format/dtype is used to open both `source_dataset_path` and `eval_query_dataset_path`, since real datasets ship their base and query files in the same layout.

**`SourceFormat.XBIN`** (`XBinReader`) — exactly the format [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/dataset_io.py)' own code calls `xbin_mmap`/`u8bin_write`:

```text
[uint32 num_vectors][uint32 dim][ num_vectors x dim raw values, row-major ]
```

One global header (once, at the start of the file), followed by the flat vector data. This isn't an unfamiliar format made up for this project — it's exactly the layout of this project's own SIFT-format base/query files, and it's simple enough that a C++ loader can read this header-plus-raw-data structure directly with a fixed-offset seek and a contiguous read. That's also why `XBinReader` takes `dtype` as a parameter (u8bin=uint8, fbin=float32, i8bin=int8, etc. — only the data type differs; the layout is identical).

**Microsoft SPACEV1B** (see the [SPTAG repository](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B)) turns out, on inspection, to use **exactly the same layout** — `vectors.bin`/`query.bin` are `[int32 count][int32 dim][int8 data...]`, and `truth.bin` is `[int32 count][int32 topk][int32 ids...][float32 dists...]`, which is structurally identical to XBIN, just with `dtype=int8` (and `truth.bin`'s structure also matches this package's own `write_groundtruth` XBIN layout exactly). In other words, **SPACEV1B needs no separate reader class** — just `open_source_reader(path, SourceFormat.XBIN, dtype=np.int8)`. `example/inspect_source_formats.py` actually reads a hand-built SPACEV-style file this way and verifies it. (Note: the real, full-scale 1.4B-vector SPACEV1B base set is additionally split across multiple physical files — `vectors_1.bin`, `vectors_2.bin`, etc. Reading that multi-file split is not implemented here; concatenate the shards into one file first if you need it.)

**`SourceFormat.VECS`** (`VecsReader`) — the original INRIA/TEXMEX format ([corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/); the original ANN_SIFT1M/1B release is in this format), commonly called fvecs (`dtype=float32`), ivecs (`dtype=int32`), or bvecs (`dtype=uint8`). **Unlike XBIN, there is no global header** — every single vector is individually prefixed with its own dimension as an int32:

```text
[int32 dim][ dim values ]   -- this record repeats num_vectors times (until the file ends)
```

So `num_vectors` isn't recorded anywhere in the file, and is instead derived as **file size ÷ record size** (`VecsReader.__init__`). fvecs/ivecs/bvecs differ only in value dtype and share an identical layout, so just as `XBinReader` covers both u8bin and fbin via `dtype`, one `VecsReader` class covers all three.

| | `SourceFormat.XBIN` | `SourceFormat.VECS` |
|---|---|---|
| Header | One per file (`num_vectors`, `dim`) | None (every record repeats its own `dim`) |
| Real-world usage | This project's own SIFT-format files, big-ann-benchmarks generally, SPACEV1B (dtype=int8) | The original corpus-texmex ANN_SIFT1M/1B, many ANN benchmark forks |
| How `num_vectors` is known | Read straight from the header | Derived as file size ÷ record size |
| Bulk batch-read performance | One `fromfile` call | Read in one shot via a structured dtype (not one call per record) |

Both readers satisfy the same structural interface (`num_vectors`/`dim`/`dtype`/`read_range`/`read_rows`/`close`, defined as `SourceVectorReader` via `typing.Protocol`), so `clustering.py`/`pool_split.py`/`groundtruth.py` never need to know which one is in use — only `StreamingOrganizerConfig.source_format` changes.

**Sources**: [big-ann-benchmarks dataset_io.py](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/dataset_io.py) · [SPTAG SPACEV1B](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) · [corpus-texmex (Yael file format)](http://yael.gforge.inria.fr/file_format.html)

### Dataset layout: how the raw file gets cut into three pools

First, a picture of what physically happens (the section after this re-explains each step at the code level).

```text
raw base dataset file (N rows, e.g. N=1,000,000,000 for a SIFT1B-scale dataset)
┌──────────────────────────────────────────────────────────────────────┐
│ row 0, 1, 2, 3, ..................................................N-1│
└──────────────────────────────────────────────────────────────────────┘

  1) k-means assigns a cluster label to every row (unrelated to a row's
     physical position -- rows in the same cluster can be scattered far
     apart in the file)

     cluster 0 = { row 3, row 91, row 402, ... }        (size s_0)
     cluster 1 = { row 7, row 15, row 88, ... }          (size s_1)
     ...
     cluster k-1 = { ... }                                (size s_{k-1})

  2) shuffle each cluster's rows independently, then slice from the front
     to assign base/insert/query (exact counts, apportioned proportional
     to cluster size via the largest-remainder method -- see "Pool sizes
     are controlled by exact counts" below):

     cluster c (size s_c, shuffled)
     ┌─────────────┬───────────┬─────────┬────────────────────┐
     │ base(n_b,c) │insert(n_i,c)│query(n_q,c)│    unused        │
     └─────────────┴───────────┴─────────┴────────────────────┘

  3) gather every cluster's base slice into one base_pool file, every
     cluster's insert slice into one ordered insert-row sequence, and
     every cluster's query slice into one ordered search-row sequence
     (this "query" pool is stream traffic -- see "Two kinds of queries"
     above; the separate eval_query_pool below comes from its own file,
     not from this split):

     base_pool.u8bin = sort_by_original_row_id( base(0) ∪ base(1) ∪ ... ∪ base(k-1) )
     insert_rows    = order_by(insert_order)( insert(0) ∪ insert(1) ∪ ... ∪ insert(k-1) )
     search_rows    = order_by(search_order)( query(0) ∪ query(1) ∪ ... ∪ query(k-1) )

     insert_rows/search_rows are then cut into per-step segment files, not
     written out as one flat file each -- see "Output layout: one file
     per step" below for the physical `insert/step_00001.u8bin`-style
     layout (`.u8bin` here because this walkthrough's dtype is uint8 --
     see "Output format" below for the general dtype -> extension rule),
     and "Segment locality" for how insert_order/search_order are chosen
     together.

     base_ratio+insert_ratio+query_ratio need not sum to 1.0, so there can
     always be an "unused" leftover (the fourth slot in the picture above)
     -- meaning: carve out exactly as much as needed from a huge source
     dataset and simply don't use the rest.

  4) separately, the entire eval_query_dataset_path file is copied out as
     eval_query_pool.u8bin, unchanged in row order/count -- no split, no
     clustering, since it's already a final query set.
```

### The full pipeline (code-flow view)

```text
source dataset (SourceFormat.XBIN or .VECS -- see above)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 1) KMeansClusterAssigner  (clustering.py)                      │
│                                                                │
│    ① random sampling #1 (fit sample)                            │
│       draw cluster_sample_size rows without replacement out of  │
│       N (e.g. 5,000 out of 20,000)                              │
│                       │                                        │
│                       ▼                                        │
│    ② fit k-means on that sample only (compute k centroids)      │
│                       │                                        │
│                       ▼                                        │
│    ③ assign a cluster label to ALL N rows via predict()          │
│       (only the sample is used to fit, but every row gets a     │
│        label; processed in batches to save memory)              │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
        cluster_ids[0..N)   (which of the k clusters each row belongs to)
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2) PoolSplitter  (pool_split.py)                                │
│                                                                │
│    ④ random sampling #2 (pool-assignment shuffle)                │
│       shuffle each cluster's rows independently                  │
│                       │                                        │
│                       ▼                                        │
│    ⑤ apportion "exact counts" proportional to cluster size       │
│       (num_base, num_insert, num_query are absolute values the   │
│        caller specifies, e.g. base=1,000,000 / insert=100,000 /  │
│        query=10,000). Uses the largest-remainder method so the   │
│        totals come out exact with no rounding error -- computing │
│        exactly how many rows each cluster contributes            │
│       (base first → insert from what's left → query from what's  │
│        left after that)                                          │
└──────────────────────────────────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────┬───────────────────┐
     ▼                 ▼                 ▼                   ▼
 base_rows        insert_rows       query_rows          (unused rows)
 (sorted           (reordered per     (reordered per       → not included
  ascending)        insert_order)      search_order)          in any pool
     │                 │                 │
     ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 3) PoolWriter  (formats.py) -- OutputFormat.XBIN or .NUMPY       │
│    streams in 1,000,000-row chunks (never loads the whole thing  │
│    into memory); insert_rows/query_rows are each cut into one     │
│    file per step first (see "Output layout: one file per step")   │
└──────────────────────────────────────────────────────────────┘
     │                 │                 │
     ▼                 ▼                 ▼
base_pool.u8bin  insert/step_*.u8bin search_query/step_* eval_query_dataset_path
 (or .npy)        (or .npy, 1/step)   (or .npy, 1/step)    (read whole, separately)
                                                                     │
                                                                     ▼
                                                          eval_query_pool.u8bin (or .npy)
                                                                     │
                                          ┌──────────────────────────┘
                                          ▼
                            ┌───────────────────────────────┐
                            │ 4) GroundTruthComputer          │
                            │    (groundtruth.py)              │
                            │    exhaustive (brute-force)        │
                            │    distance from every eval query  │
                            │    vector to every base_pool       │
                            │    vector → top-k                 │
                            │    runs on CPU(numpy) or GPU(cupy)  │
                            │    (see "GPU-accelerated" below)    │
                            └───────────────────────────────┘
                                          │
                                          ▼
                        groundtruth/step_00001.bin, ... (or .npz)
                                          │
                                          ▼
                                   manifest.npz
                       (paths, counts, per-cluster [start,end)
                        ranges, the seed actually used, etc. --
                        never JSON)
```

### "Random sampling" actually happens in three different places

This is a point that's easy to get confused about, so let's be explicit.

| # | Where | What gets sampled randomly | Purpose |
|---|--------|---------------------------|------|
| ① | `KMeansClusterAssigner.fit_and_assign` | `cluster_sample_size` rows drawn without replacement out of all N rows | Used only to **fit** k-means (fitting on all N rows would be too slow). Label assignment (predict) then runs on **all N rows** regardless of the sample |
| ② | `PoolSplitter.split` (per cluster) | Independently shuffles the rows belonging to each cluster | To pick, "randomly," which rows in that cluster go to base/insert/query (shuffling then slicing from the front is exactly a random draw) |
| ③ | `_derive_run_seed` (organizer.py) | Deterministically derives this generation's own seed from (base_seed, set name) | Makes ①② draw different actual values for every generation (`set_1`, `set_2`, ...) -- see "Workload Generation Sets" below |

The key point in ② is that "shuffle, then slice" *is* the random sampling: if a cluster has size `s_c` and `n_base` of it are assigned to base, the whole `s_c`-row cluster is shuffled and the first `n_base` rows are taken as-is. Since the shuffle is random, this is equivalent to "uniformly randomly drawing `n_base` rows from that cluster." (The separate `eval_query_pool` is not part of this random sampling at all -- it's read in its entirety, unchanged, from `eval_query_dataset_path`.)

### Pool sizes are controlled by exact counts (not ratios)

`StreamingOrganizerConfig.pool_sizes` takes **absolute counts**, like `PoolSizes(num_base=.., num_insert=.., num_query=..)`. The three counts need not sum to the full dataset size (N) — if they sum to less than N, the remainder is simply left out of every pool (e.g. out of a 1-billion-row dataset, use only base 1M + insert 100K + query 10K, and leave the rest completely unused for this experiment). If they sum to more than N, a `ValueError` is raised immediately.

To keep every pool's cluster composition tracking the dataset's natural cluster distribution, the exact "how many from each cluster" figure is computed proportional to that cluster's size (`_apportion_with_capacity`, the largest-remainder / Hare-quota method — leftover units from rounding are handed out one at a time starting with the cluster with the largest fractional remainder, so the total never drifts). Base is apportioned first, then insert from each cluster's remaining capacity (rows not yet claimed by that cluster), then query from what's left after that — so the three pools never overlap.

```text
Example: cluster 3 has size s_3 = 2,448, and was apportioned 979 rows as
     its "share" of a total of 8,000 for base:

     cluster_3_rows (2,448 rows, shuffled)
     ┌─────────────┬───────────┬────────┬────────────────┐
     │   base(979) │insert(245)│query(61)│   unused(1163) │
     └─────────────┴───────────┴────────┴────────────────┘
      first 979       next        next      the remainder (not included
      rows taken      245         61        in any pool)
```

### Insert-pool ordering controls locality (`PoolRowOrder`)

The **order rows are stored in** within the `insert_pool` file is exactly what determines "insert locality." A benchmark loader that only supports reading a contiguous `[start, end)` range within a file needs "insert cluster C only" to mean C's rows are **physically contiguous in the output file**.

- `PoolRowOrder.CLUSTER`: rows are grouped in ascending cluster-id order → `manifest.insert_cluster_ranges` records each cluster's `[start, end)`, which can be passed directly as the `start`/`end` parameters.
- `PoolRowOrder.CLUSTER_SHUFFLED`: same per-cluster contiguous grouping as `CLUSTER` (so `insert_cluster_ranges` is still populated), but the clusters are visited in a random permutation instead of ascending id order. Used to give a stream real per-segment locality without that locality being synchronized to another stream's cluster sequence — see `SegmentLocality.NONALIGN` below.
- `PoolRowOrder.NATURAL`: kept in the original dataset's row order. Useful as a regression-check baseline, since it always produces a single, fixed insert order.
- `PoolRowOrder.RANDOM`: fully shuffled. A locality-free control condition.

The query (search) pool is ordered the same way, via its own independent `search_order` value — see "Segment locality" below for how `insert_order`/`search_order` are chosen together rather than as two free-standing choices.

### Output format is a choice of two (`OutputFormat`) — JSON is used nowhere

`StreamingOrganizerConfig.output_format` picks, in one shot, how pools, ground truth, and the manifest are all serialized.

| | `OutputFormat.XBIN` (recommended default) | `OutputFormat.NUMPY` |
|---|---|---|
| Pool files | `base_pool<ext>`, `insert/step_00001<ext>`, etc. -- the same header+raw-bytes layout as the source (`[int32 N][int32 dim]` header + raw bytes), where `<ext>` is the **dtype-specific** extension big-ann-benchmarks/cuVS use for vector files, not a fixed `.xbin` (see table below) | `base_pool.npy`, `insert/step_00001.npy`, etc. -- the standard numpy `.npy`, regardless of dtype |
| Groundtruth file | `groundtruth/step_00001.bin`, ... (one per checkpoint) -- `(uint32 n)(uint32 k)` header + int32 id matrix + float32 distance matrix (same convention as bigann's `gt100.bin`/SPACEV1B's `truth.bin`; always this fixed shape regardless of the pool dtype) | `groundtruth/step_00001.npz`, ..., a compressed archive holding `neighbor_ids`/`neighbor_dists` |
| Use case | **a simple, fixed-header C++ loader can read it directly, with zero code changes** — for actually running a benchmark | Convenient to quickly inspect (`np.load`) or further process from Python — needs a separate converter on the C++ side |

In both cases, pool files are streamed in 1,000,000-row chunks (`PoolWriter`), so no matter how large a pool is, the whole thing is never loaded into memory (the NUMPY path pre-allocates the file on disk via `numpy.lib.format.open_memmap` and fills it in, streaming exactly like XBIN).

`<ext>` (`OutputFormat.pool_file_suffix(dtype)`, `formats.py`) is resolved from the pool's actual vector dtype, matching the same filename convention real datasets on disk already use (e.g. this project's own `base.1B.u8bin`) and that [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) / [cuVS's benchmarking guide](https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets) document:

| dtype | extension |
|---|---|
| `uint8` | `.u8bin` |
| `int8` | `.i8bin` |
| `float16` | `.f16bin` |
| `float32` | `.fbin` |

This is purely a filename convention — the on-disk header+raw-data layout is identical for every dtype (see `XBinReader`/`XBinWriter` above). The `[dataset] dtype = ...` ini setting (or `StreamingOrganizerConfig.dtype`) is what actually drives this: it's still an explicit, required setting, never inferred from a filename or auto-detected.

The `manifest` (path/count/cluster-range/seed metadata) is also stored not as JSON but as **`manifest.npz`** (a numpy compressed archive). No text/JSON parsing is used anywhere in this package — everything is always serialized as numpy-native binary.

Given that another (non-SIFT) dataset might need support, or the output format itself might need to change again, `OutputFormat` is implemented as a **closed enum plus one branch**, not an open class hierarchy: adding a new format is just one new enum value plus one new branch in `PoolWriter`/`write_groundtruth` (no new subclass needed).

---

## GPU-Accelerated Ground Truth (`ComputeDevice`)

`GroundTruthComputer`'s exhaustive kNN (distance from every query to every base vector, then top-k) can run on **GPU (cupy)** as well as CPU (numpy).

```python
from arachne.workload import ComputeDevice, GroundTruthComputer, DistanceMetric

computer = GroundTruthComputer(
    k=100, metric=DistanceMetric.EUCLIDEAN,
    device=ComputeDevice.GPU,   # or ComputeDevice.CPU (the default)
)
```

This is passed straight through to `StreamingWorkloadOrganizer` via `StreamingOrganizerConfig.groundtruth_device` (see `example/generate_streaming_workload.py`, which actually detects whether GPU is available and picks accordingly).

### Why this works

This environment already runs on a Blackwell GPU with a CUDA 12.8 + RAFT/RMM stack, and `cupy` (a library exposing a nearly numpy-identical API for GPU arrays) is **already installed** as a transitive dependency of that stack (used internally by `pylibraft`/`rmm`). `cuvs` (RAFT's full ANN Python bindings) and `pylibraft.neighbors` (which would provide GPU brute-force kNN directly) aren't available in this environment, so that route isn't usable — but cupy alone lets the existing numpy code run on GPU **almost unchanged**: the `_pairwise_distance`/top-k selection logic shares both backends by swapping a single array-module variable, `xp = np or cp` (no new class needed).

### How it's implemented

- `ComputeDevice.CPU` (default): computes with numpy, same as before.
- `ComputeDevice.GPU`: moves the query vectors and each base batch to GPU memory (`cupy.asarray`), runs the distance computation + top-k merge on GPU, and copies only the final result back to host (`.get()`). **`GroundTruthResult` always holds plain numpy arrays**, so the rest of the pipeline (`formats.py`'s `write_groundtruth`, etc.) never needs to care which device did the work.
- If `ComputeDevice.GPU` is used in an environment without `cupy` installed, a `RuntimeError` is raised immediately **at construction time**, before any computation starts (much easier to debug than failing partway through).

### Verified

CPU and GPU paths were directly cross-checked to produce identical results (both Euclidean and inner-product metrics: neighbor ids match exactly, and distances match within `atol=1e-2` — the tiny remaining discrepancy is just floating-point operation-order differences). GPU is expected to be meaningfully faster for large base pools, but no dedicated benchmark numbers have been measured yet.

---

## Workload Generation Sets (`set_1`, `set_2`, ...)

Every call to `StreamingWorkloadOrganizer.run()` is **one "generation."** Each one lands in its own subfolder, so running the same `StreamingOrganizerConfig` multiple times keeps accumulating comparable sets instead of overwriting each other.

```text
output_root/
  set_1/
    base_pool.u8bin
    insert/          search_query/      delete/            groundtruth/
      step_*.u8bin     step_*.u8bin       step_*.ids          step_*.bin
    eval_query_pool.u8bin
    manifest.npz
  set_2/                     <- created automatically by calling run() again with the same StreamingOrganizerConfig
    base_pool.u8bin           (a different random sample than set_1! see below)
    ...
  set_3/
    ...
```

(see "Output layout: one file per step" below for exactly what goes in `insert/`/`search_query/`/`delete/`/`groundtruth/`.)

- If `set_name` isn't given, the existing `set_*` folders under `output_root` are scanned and the **next number is picked automatically** (starting at `set_1` if none exist).
- Running with a name that already exists (e.g. explicitly repeating `set_name="set_1"`) **raises an exception and overwrites nothing.**
- If a generation fails partway through (e.g. requested pool sizes exceed the dataset), the half-built folder is **automatically removed** — so a failed set never looks like a completed one, or confuses the auto-numbering.
- How many sets to generate in one go is a run-level choice, not baked into `StreamingOrganizerConfig` — see `RunSettings.num_sets` in "Config files (`.ini`)" below.

### Why `set_2` ends up with a different sample than `set_1`

`StreamingOrganizerConfig.random_seed` is only a **base seed** — it isn't the seed actually used for clustering/shuffling. `run()` deterministically derives **a seed specific to this set** from the `(base_seed, set_name)` pair every time (`_derive_run_seed`, crc32 hash + `numpy.random.SeedSequence`). As a result:

- Calling `run()` twice with the same `StreamingOrganizerConfig` makes `set_1` and `set_2` draw **different** random samples.
- **Reproducibility is still guaranteed**: the pair `(random_seed=42, set_name="set_1")` always produces the same result no matter when it's run again. The seed actually used for a given generation is recorded in `manifest.random_seed`.

---

## Logging

Every module reports status through a child of the shared `arachne.workload` logger (`logging_utils.py`). Library code never attaches a handler itself, so turn it on once at the top of your script.

```python
import logging
from arachne.workload import configure_logging

configure_logging(level=logging.INFO)   # or logging.DEBUG for more detail
```

At `INFO`, you see pipeline-stage transitions (clustering start/finish, pool sizes, file paths, ground-truth progress); at `DEBUG`, you also see per-cluster apportionment detail and per-batch progress. Example:

```text
2026-07-06 17:09:31 [INFO] arachne.workload.clustering: fitting k-means: k=8 sample_size=5000 (of 20000 total rows)
2026-07-06 17:09:31 [INFO] arachne.workload.clustering: cluster assignment complete: sizes min=2387 max=2601 mean=2500.0
2026-07-06 17:09:31 [INFO] arachne.workload.pool_split: splitting 20000 rows across 8 clusters into base=8000 insert=2000 query=500 (unused=9500)
2026-07-06 17:09:31 [INFO] arachne.workload.organizer: writing base pool (8000 rows) -> .../set_1/base_pool.u8bin
2026-07-06 17:09:31 [INFO] arachne.workload.groundtruth: computing exhaustive groundtruth on cpu: num_queries=500 num_base=8000 k=10 metric=euclidean
2026-07-06 17:09:31 [INFO] arachne.workload.organizer: === streaming workload generation complete: set=set_1 ===
```

---

## Streaming workloads: insert(+delete)+search (`StreamingWorkloadOrganizer`)

A single static ground-truth file only answers "what are the true top-k neighbors before any insert has happened" — it can't measure recall *while* a stream of inserts (and deletes) is running, since the true top-k neighbors change as vectors arrive and expire. `StreamingWorkloadOrganizer` (`streaming.py` + `organizer.py`) is the part of the pipeline that handles this: it turns the insert pool and the query (search) pool into an ordered sequence of per-step segments, and computes one ground-truth checkpoint per step against whichever rows are actually active at that point.

### Two independent workload kinds (`WorkloadKind`)

Two independent workload kinds are built on top of the same insert/search-as-ordered-steps model:

- **`WorkloadKind.INSERT_SEARCH`** — pure growth. Every step inserts one segment; nothing is ever deleted (`StreamingConfig.num_delete` must be `0`).
- **`WorkloadKind.INSERT_DELETE_SEARCH`** — every step inserts one segment and also deletes its share of `num_delete` rows (see "`StreamingConfig`" below), chosen **uniformly at random** from whatever is active immediately before that step's own insert — base-pool rows are exactly as eligible as any already-inserted row, and there is no FIFO/oldest-first rule at all.

These are kept as two separate configs (not folded into one "delete probability" knob) because they answer different questions: INSERT_SEARCH isolates recall drift under pure growth, while INSERT_DELETE_SEARCH additionally exercises the reclaim/eviction path. Uniform-random deletion (independent of insertion order) is one point on a broader axis of possible delete policies — *how correlated deletion order is with insertion order* — that also includes FIFO/oldest-first (fully correlated) and per-vector lifespan-based expiry; only the uniform-random policy is implemented today (see "What's still open" below).

### Segment locality (`SegmentLocality`): does step *i*'s insert land where step *i*'s search looks?

`WorkloadKind` controls *when* vectors come and go; `SegmentLocality` is a separate, orthogonal axis controlling *where in the cluster space* each step's insert segment and search segment fall, relative to each other. Both the insert stream and the search (stream-query) stream are cut from the same clustered dataset and can each independently be ordered by cluster or scattered. `PoolSplitter.split()`'s `query_order` parameter allows any combination, but most of the full `PoolRowOrder` x `PoolRowOrder` cross product isn't actually useful to distinguish for this purpose (`NATURAL` and plain `RANDOM` both mean "no per-segment cluster concentration" here regardless of exactly how they scatter, and pairing a scattered insert stream with a cluster-ordered search stream gives insert no "hot" cluster for search to align with or diverge from in the first place). That leaves exactly three combinations worth naming, and `SegmentLocality` (streaming.py) is exactly those three:

| `SegmentLocality` | `insert_order` | `search_order` | What happens |
|---|---|---|---|
| `ALIGN` | `CLUSTER` | `CLUSTER` | Both streams grouped by cluster **in the same ascending cluster-id sequence**, so step *i*'s insert segment and step *i*'s search segment tend to draw from the *same* cluster region. High per-segment insert/search locality, both drifting through clusters together. |
| `NONALIGN` | `CLUSTER` | `CLUSTER_SHUFFLED` | Insert keeps its ascending cluster-ordered drift. Search is *also* grouped by cluster (so each search segment still has real intra-segment locality -- it isn't scattered), but the sequence clusters appear in is an **independent random permutation**, decorrelated from insert's ascending sequence. Segment *i* still means "mostly one cluster" for search, just not (usually) the same one insert is on. |
| `RANDOM` | `RANDOM` | `RANDOM` | Neither stream is grouped by cluster at all -- every position in every segment is an equally likely draw from any cluster. Neither drifts, and neither has intra-segment locality either; a flat baseline. |

The key distinction NONALIGN is built to capture: it is **not** "search has no locality" -- that's what `RANDOM` is for. It's "search *does* drift through clusters one segment at a time, just on a schedule uncorrelated with insert's." This needed a fourth `PoolRowOrder` value, `CLUSTER_SHUFFLED` (pool_split.py): identical to `CLUSTER`'s per-cluster contiguous grouping (so `search_cluster_ranges` still comes back populated, not empty), except the clusters are visited in a random permutation instead of ascending id order.

```python
from arachne.workload import SegmentLocality

SegmentLocality.ALIGN.search_order      # PoolRowOrder.CLUSTER
SegmentLocality.NONALIGN.search_order   # PoolRowOrder.CLUSTER_SHUFFLED  (insert_order is still PoolRowOrder.CLUSTER)
SegmentLocality.RANDOM.search_order     # PoolRowOrder.RANDOM
```

`StreamingOrganizerConfig.segment_locality` takes the place of the plain `PoolRowOrder` field `OrganizerConfig` has, since for a streaming run the insert and search orderings need to be picked *together* (see table above) rather than as two independently free-standing choices. `StreamingWorkloadManifest` records both the resolved `insert_order`/`search_order` values (so a consumer never needs the `SegmentLocality` → `(insert_order, search_order)` mapping itself to know how to interpret `insert_cluster_ranges`/`search_cluster_ranges` — empty under `RANDOM`, populated under `CLUSTER`/`CLUSTER_SHUFFLED`) and `segment_locality` itself, for whichever a downstream reader finds more convenient.

This was verified directly on a 16-cluster synthetic dataset split into 10 steps: under `ALIGN`, insert and search land in the same 1-2 clusters at every step (e.g. step 4: insert `{4,5,6}`, search `{4,5,6}`). Under `NONALIGN`, insert keeps that same drift, but search's clusters are a different, shuffled set at every step (step 4: insert `{4,5,6}`, search `{3,11,12}`) -- `search_cluster_ranges` is still populated (not empty), confirming search retains real per-segment locality rather than being scattered. Under `RANDOM`, both `insert_cluster_ranges` and `search_cluster_ranges` come back empty (fully scattered on both sides).

### `StreamingConfig` + `PoolSizes`: the "iteration set" and the insert:search:delete rates

```python
from arachne.workload import PoolSizes, StreamingConfig, WorkloadKind

pool_sizes = PoolSizes(
    num_base=1_000_000,
    num_insert=1_000_000,   # total insert-pool rows across the whole run
    num_query=500_000,      # total search-pool rows across the whole run -> a 2:1 insert:search rate
)
streaming_config = StreamingConfig(
    workload_kind=WorkloadKind.INSERT_DELETE_SEARCH,
    num_steps=200,          # the "iteration set" size: how many insert(+delete) steps this run has
    num_delete=500_000,     # total rows deleted across the whole run (must be 0 for INSERT_SEARCH)
    checkpoint_every=1,     # 1 = a groundtruth checkpoint at every step
)
```

`num_insert`/`num_query`/`num_delete` are each a *total* for the whole run, not a per-step size — `num_steps` divides each one into that many near-equal per-step pieces (`streaming._split_into_steps`: `total // num_steps`, with the remainder handed one-each to the first `total % num_steps` steps, so no step's size ever differs from another's by more than 1). This means the insert:search rate is simply whatever ratio `num_insert:num_query` is chosen in (1,000,000:500,000 above is 2:1) — there's no separate per-step size to keep in sync with those totals, and no rounding/apportionment scheme to design for an explicit "rate" field.

`num_delete` deletes are **not** a fixed range: each step's share is chosen *uniformly at random* from whatever is active immediately before that step's own insert (base rows and every not-yet-deleted previously-inserted row alike). `StreamingPlan` validates this schedule eagerly — purely from `num_base`/`num_insert`/`num_delete`/`num_steps`, no dataset or RNG needed — by tracking, step by step, how many rows *could* be active at that point, and raising immediately if any step would be asked to delete more than that. This runs the moment a `StreamingOrganizerConfig` is constructed (see its `__post_init__`), so a schedule that's too aggressive (e.g. `num_delete` too large relative to `num_base`+early `num_insert` shares) fails before the source dataset is even opened, let alone clustered.

### Output layout: one file per step

Unlike the base pool and the evaluation query pool (each a single file — base is a one-time bulk build, eval queries are the same fixed set graded at every checkpoint), the insert stream and the search-query stream are each written as **one file per step**, plus one id-list file per step where a delete happens:

```text
set_1/
  base_pool.u8bin
  eval_query_pool.u8bin
  insert/
    step_00001.u8bin  step_00002.u8bin  ...   step_00200.u8bin
  search_query/
    step_00001.u8bin  step_00002.u8bin  ...   step_00200.u8bin
  delete/
    step_00001.ids    step_00002.ids    ...   step_00200.ids   (only steps with a non-zero delete share)
  groundtruth/
    step_00001.bin    step_00002.bin    ...   step_00200.bin   (only checkpoint steps)
  manifest.npz
```

(`.u8bin` above because this walkthrough's dtype is uint8 -- a `float32` run would instead get `.fbin` pool files; see "Output format" above.) The naming convention (`step_{i:05d}<suffix>`, 1-indexed, shared by all four directories) is the single fact a consumer needs to reconstruct every path — no manifest lookup required beyond knowing `num_steps` (and, for `groundtruth`, `checkpoint_steps`, itself a plain scalar list). `ls insert/`/`ls search_query/` answers "how many steps" on its own; each file's own xbin/npy header answers "how big is this step's segment" (sizes can differ by at most 1 row from step to step, per `_split_into_steps` above, so they aren't computable by a fixed multiply -- reading the header is cheap and exact).

`delete/step_{i:05d}<id_list_file_suffix>` holds only **global ids**, not vector data (`write_id_list`/`read_id_list`, formats.py) — sorted, and *not* necessarily contiguous: `0..base_pool_count-1` is a position in the base pool, `base_pool_count + j` is position `j` in the insert stream. A consumer already received the corresponding vectors when they were inserted (or in the base build), so re-writing them here would just be a redundant copy; only the reference is needed to know what to remove.

### Step timeline (ASCII diagram)

```text
PoolSizes(num_base=10, num_insert=12, num_query=6)
StreamingConfig(workload_kind=INSERT_DELETE_SEARCH, num_steps=6, num_delete=6, checkpoint_every=1)

insert/            step_00001  step_00002  step_00003  step_00004  step_00005  step_00006
search_query/      step_00001  step_00002  step_00003  step_00004  step_00005  step_00006

step:                   1           2           3           4           5           6
insert count:           2           2           2           2           2           2     (12 / 6 steps)
delete count:           1           1           1           1           1           1     (6 / 6 steps -- uniformly at random each time, not FIFO)
delete/ file:      step_00001  step_00002  step_00003  step_00004  step_00005  step_00006  (every step here, since 6/6 is never 0)
active count:          11          12          13          14          15          16     (10 base, net +1/step)
groundtruth/ file: step_00001  step_00002  step_00003  step_00004  step_00005  step_00006  (checkpoint_every=1)
```

For `WorkloadKind.INSERT_SEARCH`, `num_delete` is always `0`, so there is no `delete/` file at any step — the active set simply keeps growing by each step's insert count. If `num_delete` doesn't divide evenly into `num_steps`, the remainder goes to the *first* few steps (`_split_into_steps`), so it's possible for later steps to have a delete count of `0` (no file) even under `INSERT_DELETE_SEARCH`.

Insert/search counts (and therefore `insert_range(step)`/`search_range(step)`) are closed-form arithmetic, precomputed once by `StreamingPlan` from `(num_steps, num_insert, num_query)` -- no per-step log needed. *Which* rows get deleted is not closed-form (only the *count* is, via `delete_count(step)`): it depends on the actual random draw at every earlier step, so `StreamingWorkloadOrganizer.run()` replays the steps in order, keeping a running Python `set` of active global ids that it draws each step's deletions from and folds each step's insertions into.

### Ground truth per checkpoint

A single static ground-truth file would only be correct for one step, so `StreamingWorkloadOrganizer` computes **one groundtruth file per checkpoint step** (`streaming_config.checkpoint_steps`), each scored against whichever rows are actually active immediately after that step's delete-then-insert, against the same `eval_query_pool` at every checkpoint (`compute_checkpoint_groundtruth`).

"Active" is an explicit, generally non-contiguous set of global ids (not a range, since deletion is random) -- position `0..base_pool_count-1` is a position in the base pool, `base_pool_count + j` is position `j` in the insert stream. `compute_checkpoint_groundtruth` translates this id array into actual source-dataset rows and calls the same `GroundTruthComputer.compute()` used everywhere else, unchanged. A GT result's position `i` corresponds to `active_ids[i]`, not to a fixed row range -- but since `active_ids` is fully determined by which rows have been inserted/deleted so far (both recorded exactly in the `insert/`/`delete/` files), a downstream consumer that independently replays those files and sorts its own active-id bookkeeping the same way (ascending global id) arrives at the identical array, and so can match GT positions to its own vectors without arachne needing to persist this array separately.

With `checkpoint_every=1` a `num_steps=200` run computes 200 full exhaustive-scan ground truths — GPU acceleration (`ComputeDevice.GPU`, above) matters a lot here; raise `checkpoint_every` to trade checkpoint resolution for less recomputation cost.

### `StreamingWorkloadManifest`

- `base_pool_path`, `base_pool_count` — the one-time bulk base build (never segmented).
- `segment_locality` — the resolved `SegmentLocality` (`"align"`/`"nonalign"`/`"random"`) this run used.
- `insert_dir`, `insert_pool_count`, `insert_order`, `insert_cluster_ranges` — the insert stream's directory and per-cluster ranges (in *global* insert-stream position, not per-file). `insert_order` is the concrete `PoolRowOrder` `segment_locality` resolved to.
- `search_dir`, `search_pool_count`, `search_order`, `search_cluster_ranges` — the search-query stream, same shape as the insert fields.
- `delete_dir` — directory of per-step id-list files (empty if nothing is ever deleted).
- `eval_query_dataset_path`, `eval_query_pool_path`, `eval_query_pool_count` — the separate, already-final query set graded at every checkpoint.
- `workload_kind`, `num_steps`, `num_delete`, `checkpoint_every`, `checkpoint_steps` — the resolved `StreamingConfig` (`insert_pool_count`/`search_pool_count` above already give the insert/search totals -- there's no separate `num_insert`/`num_query` field here).
- `groundtruth_dir` — directory of per-checkpoint groundtruth files.
- `random_seed` — the derived per-set seed actually used (see "Workload Generation Sets" above).

Serialized as `manifest.npz`, no JSON.

### What's still open

Only uniform-random deletion (independent of both insertion order and cluster) is implemented. Other delete policies are worth considering for the future: FIFO/oldest-first (deletion order fully correlated with insertion order -- an earlier design of this pipeline actually implemented this, via a fixed-size sliding window, before being replaced by the uniform-random policy above), per-vector lifespan-based expiry (each vector assigned a random or class-based lifetime at insert time), and round-based per-cluster insert/delete cycling (which would connect most directly to `arachne.workload`'s own cluster labels for locality-controlled experiments). Each would need its own random-draw rule in place of the uniform sample `StreamingWorkloadOrganizer.run()` takes today.

---

## Config files (`.ini`)

`StreamingOrganizerConfig`, plus the run-level settings around it (which files to read, where to write, how many sets to generate), can be loaded from a plain `.ini` file instead of being constructed in Python — so a workload can be fully described in one text file and handed to a script, without editing Python source for every new dataset/pool-size/streaming shape.

```python
from pathlib import Path
from arachne.workload import load_streaming_organizer_config

streaming_config, run_settings = load_streaming_organizer_config(Path("my_streaming_workload.ini"))
# streaming_config: StreamingOrganizerConfig
# run_settings: RunSettings(source_dataset_path, eval_query_dataset_path, output_root, num_sets)
```

`RunSettings` holds exactly what isn't part of `StreamingOrganizerConfig` itself: `source_dataset_path`, `eval_query_dataset_path`, `output_root`, and `num_sets` (how many `set_N` generations to produce in a loop) — mirroring how `source_dataset_path`/`output_root` are already `run()` arguments, not workload-shape knobs.

See `workload/example/streaming_workload.example.ini` for a complete, annotated template covering every section (`[dataset]`, `[clustering]`, `[pools]`, `[streaming]`, `[groundtruth]`, `[output]`) with placeholder paths meant to be copied and pointed at a real dataset.

For an ini file you can actually run immediately (no real dataset needed), see `workload/example/quickstart_streaming_workload.ini` — it points at the same synthetic files the example script's no-argument demo mode already writes to `workload/example/output/`:

```bash
cd arachne   # top level of this repository
PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py           # 1) writes the synthetic files first
PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py \
    --config python/arachne/workload/example/quickstart_streaming_workload.ini                     # 2) reruns the same workload, but driven entirely by the ini file
```

`[pools] num_base`/`num_insert`/`num_search` set the workload's ratios directly and must always be given explicitly (there's no per-step size to derive them from anymore); `[pools] num_delete` alone may be left blank (-> `0`, no deletes).

This uses only the standard library's `configparser` — no new dependency, consistent with the rest of this package's "no third-party serialization format" stance (see "Output format" above).

---

## Usage example

`workload/example/generate_streaming_workload.py` is the full runnable example; below is just the essence of it.

```python
import logging
from pathlib import Path

import numpy as np
from arachne.workload import (
    ComputeDevice, DistanceMetric, OutputFormat, PoolSizes, SegmentLocality, SourceFormat,
    StreamingConfig, StreamingOrganizerConfig, StreamingWorkloadOrganizer, WorkloadKind,
    configure_logging,
)

configure_logging(level=logging.INFO)

streaming_config = StreamingConfig(
    workload_kind=WorkloadKind.INSERT_DELETE_SEARCH,  # or INSERT_SEARCH for pure growth
    num_steps=200,                # the "iteration set" size
    num_delete=500_000,           # total rows deleted over the run; must be 0 for INSERT_SEARCH
    checkpoint_every=1,           # one groundtruth checkpoint per step
)

config = StreamingOrganizerConfig(
    source_format=SourceFormat.XBIN,
    dim=128,
    dtype=np.uint8,
    distance_metric=DistanceMetric.EUCLIDEAN,
    num_clusters=64,
    cluster_sample_size=200_000,
    pool_sizes=PoolSizes(
        num_base=1_000_000,
        num_insert=1_000_000,    # total insert-pool rows over the run, divided into num_steps segments
        num_query=500_000,       # total search-pool rows, likewise -- a 2:1 insert:search rate
    ),
    segment_locality=SegmentLocality.ALIGN,  # or NONALIGN / RANDOM -- see "Segment locality" above
    streaming_config=streaming_config,
    groundtruth_k=100,
    groundtruth_device=ComputeDevice.GPU,
    output_format=OutputFormat.XBIN,
    random_seed=42,
)

manifest = StreamingWorkloadOrganizer(config).run(
    source_dataset_path=Path("/path/to/dataset/base.1B.u8bin"),
    eval_query_dataset_path=Path("/path/to/dataset/query.public.10K.u8bin"),
    output_root=Path("/path/to/output"),
)
print(manifest.insert_dir, manifest.search_dir, manifest.groundtruth_dir)
print(manifest.checkpoint_steps[:5])
```

Or load the config from an `.ini` file instead (see "Config files (`.ini`)" above):

```python
from pathlib import Path
from arachne.workload import StreamingWorkloadOrganizer, load_streaming_organizer_config

config, run_settings = load_streaming_organizer_config(Path("my_streaming_workload.ini"))
organizer = StreamingWorkloadOrganizer(config)
for _ in range(run_settings.num_sets):
    organizer.run(
        source_dataset_path=run_settings.source_dataset_path,
        eval_query_dataset_path=run_settings.eval_query_dataset_path,
        output_root=run_settings.output_root,
    )
```

---

## Extending to another input/output format

- **To add a third input format** (there are currently two, XBIN/VECS): write a new reader class satisfying `formats.py`'s `SourceVectorReader` Protocol (`read_range`/`read_rows`/`num_vectors`/`dim`/`dtype`/`close`), add a value to `SourceFormat`, and add one branch to `open_source_reader()`. `clustering.py`/`pool_split.py`/`groundtruth.py`/`organizer.py` depend only on that Protocol (they never know the concrete class), so they're reused unchanged.
- **To add another output format**: add a value to `formats.py`'s `OutputFormat`, and add one branch each to `PoolWriter.__init__`/`write_groundtruth`.
