"""arachne.workload.organizer

Top-level orchestration: raw dataset -> cluster labels -> base/insert/query
3-pool split -> per-step pool files + ground truth + manifest. This is the
offline preprocessing tool that turns one raw dataset into a ready-to-run
insert(+delete)+search streaming ANN benchmark workload
(StreamingWorkloadOrganizer), built on the shared cluster-labeling/pool-split
building blocks below (module-level helpers, not a base class).

This module does not touch any downstream C++/CUDA ANN library source
code: with OutputFormat.XBIN (the default choice for feeding a benchmark
run) its only job is to produce xbin-format files that a simple
fixed-header, contiguous-range C++ loader can already read unmodified.

StreamingWorkloadOrganizer takes **two** separate raw inputs, not one:
`source_dataset_path` (the base dataset the base/insert/"stream query"
pools are cut from) and `eval_query_dataset_path` (a separate,
already-final query set -- e.g. a dataset's own held-out query file --
used purely to grade ground truth). See python/arachne/README.md ("Two
kinds of queries") for why these are kept separate.

Each call to run() is one "workload generation": it creates a new,
numbered `set_<N>` subdirectory under `output_root` (or a caller-chosen
name) and writes that generation's pools/groundtruth/manifest there, so
repeated generations accumulate as distinct, directly comparable sets
(set_1, set_2, ...) instead of overwriting each other. See
python/arachne/README.md for the full pipeline diagram and rationale.
"""

from __future__ import annotations

import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arachne.workload.clustering import ClusterAssignment, KMeansClusterAssigner
from arachne.workload.formats import (
    OutputFormat,
    PoolWriter,
    SourceFormat,
    SourceVectorReader,
    open_source_reader,
    write_groundtruth,
    write_id_list,
)
from arachne.workload.groundtruth import ComputeDevice, DistanceMetric, GroundTruthComputer
from arachne.workload.logging_utils import get_logger
from arachne.workload.manifest import ClusterRangeInfo, StreamingWorkloadManifest
from arachne.workload.pool_split import PoolRowOrder, PoolSizes, PoolSplit, PoolSplitter
from arachne.workload.streaming import (
    SegmentLocality,
    StreamingConfig,
    StreamingPlan,
    compute_checkpoint_groundtruth,
)

logger = get_logger(__name__)

_SET_DIR_PREFIX = "set_"


# --- Helpers used by StreamingWorkloadOrganizer.run() -----------------------
# Plain module-level functions (not methods) so each one stays independently
# readable/testable rather than living only as a step inside one large run().


def _next_set_name(output_root: Path) -> str:
    """Finds the next unused `set_<N>` name under `output_root` by
    scanning existing `set_*` subdirectories and taking one past the
    highest index found (starting at set_1 if none exist)."""
    existing_indices = []
    for child in output_root.iterdir():
        if child.is_dir() and child.name.startswith(_SET_DIR_PREFIX):
            suffix = child.name[len(_SET_DIR_PREFIX):]
            if suffix.isdigit():
                existing_indices.append(int(suffix))
    next_index = max(existing_indices, default=0) + 1
    return f"{_SET_DIR_PREFIX}{next_index}"


def _derive_run_seed(base_seed: int, set_name: str) -> int:
    """Derives a per-set random seed from a config's base seed and the
    set's name, so different sets (set_1, set_2, ...) draw different
    random samples by default while remaining fully reproducible: the
    same (base_seed, set_name) pair always derives the same seed. Uses
    crc32 (not Python's salted str hash()) so the derivation is stable
    across processes/runs.
    """
    name_hash = zlib.crc32(set_name.encode("utf-8"))
    seed_sequence = np.random.SeedSequence([base_seed, name_hash])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _assign_clusters(
    reader: SourceVectorReader, num_clusters: int, cluster_sample_size: int, run_seed: int
) -> ClusterAssignment:
    assigner = KMeansClusterAssigner(
        num_clusters=num_clusters, sample_size=cluster_sample_size, random_seed=run_seed
    )
    return assigner.fit_and_assign(reader)


def _split_pools(
    assignment: ClusterAssignment,
    pool_sizes: PoolSizes,
    insert_order: PoolRowOrder,
    run_seed: int,
    query_order: PoolRowOrder,
) -> PoolSplit:
    splitter = PoolSplitter(pool_sizes=pool_sizes, random_seed=run_seed)
    return splitter.split(assignment, insert_order=insert_order, query_order=query_order)


def step_file_path(directory: Path, step: int, suffix: str) -> Path:
    """Shared naming convention for every per-step file
    StreamingWorkloadOrganizer writes: `step_{step:05d}<suffix>`
    (1-indexed). See StreamingWorkloadManifest for which directories use
    this -- a consumer that knows `num_steps` (and, for groundtruth,
    `checkpoint_steps`) can construct every path without reading anything
    but the manifest's scalars (except `delete/`, whose files exist only
    for steps that actually delete something -- see
    StreamingWorkloadManifest)."""
    return directory / f"step_{step:05d}{suffix}"


def _write_pool(
    reader: SourceVectorReader,
    row_indices: np.ndarray,
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Streams `row_indices` (in the exact order given) from `reader`
    into a new pool file, chunked so an arbitrarily large pool never
    needs to be held in memory all at once."""
    chunk_size = 1_000_000
    with PoolWriter(
        output_path,
        output_format=output_format,
        dim=reader.dim,
        dtype=reader.dtype,
        num_vectors=row_indices.shape[0],
    ) as writer:
        for chunk_start in range(0, row_indices.shape[0], chunk_size):
            chunk_rows = row_indices[chunk_start : chunk_start + chunk_size]
            writer.write_rows(reader.read_rows(chunk_rows))


def _resolve_output_dir(output_root: Path, set_name: str | None) -> tuple[str, Path]:
    """Resolves and creates this run's `output_root/<set_name>/` output
    directory (auto-numbered if `set_name` is not given). Raises if the
    resolved directory already exists, so a generation never silently
    overwrites another."""
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_set_name = set_name if set_name is not None else _next_set_name(output_root)
    output_dir = output_root / resolved_set_name
    if output_dir.exists():
        raise FileExistsError(f"output set directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    return resolved_set_name, output_dir


def _write_eval_query_pool(
    eval_reader: SourceVectorReader,
    output_dir: Path,
    output_format: OutputFormat,
) -> tuple[Path, np.ndarray]:
    """Writes the entire eval-query dataset out as `eval_query_pool.*` (so
    the run's output directory is self-contained, just like base/insert/
    query), and returns its path plus the vectors themselves (used
    directly by GroundTruthComputer -- reading them back from disk again
    would be redundant since they were just read to write this file).
    """
    eval_query_pool_path = output_dir / f"eval_query_pool{output_format.pool_file_suffix(eval_reader.dtype)}"
    all_rows = np.arange(eval_reader.num_vectors)
    logger.info(
        "writing eval query pool (%d rows) -> %s", eval_reader.num_vectors, eval_query_pool_path
    )
    _write_pool(eval_reader, all_rows, eval_query_pool_path, output_format)
    eval_query_vectors = eval_reader.read_range(0, eval_reader.num_vectors)
    return eval_query_pool_path, eval_query_vectors


@dataclass(frozen=True)
class StreamingOrganizerConfig:
    """All knobs for one end-to-end StreamingWorkloadOrganizer run.

    `pool_sizes` (num_base/num_insert/num_query -- num_query is the
    *search* pool's total row count) sets the workload's ratios directly:
    `streaming_config.num_steps` just divides `num_insert`/`num_query`
    (and `num_delete`) into that many near-equal per-step pieces (see
    StreamingPlan) -- there is no separate per-step size knob to keep in
    sync with these totals. `pool_sizes.total` (base+insert+query) may be
    less than the source dataset's row count (the remainder is simply
    unused), but not more -- StreamingWorkloadOrganizer.run() checks this
    immediately after opening the source dataset, before the (expensive)
    clustering step.

    `segment_locality` (streaming.SegmentLocality) picks a plain
    `PoolRowOrder` for the insert stream and one for the search stream
    *together*, since the only combinations
    that make sense to distinguish for a streaming run are "both drift
    together" (ALIGN), "insert drifts, search doesn't correlate"
    (NONALIGN), and "neither drifts" (RANDOM) -- see SegmentLocality's
    docstring for why the other PoolRowOrder x PoolRowOrder combinations
    aren't offered.
    """

    source_format: SourceFormat
    dim: int
    dtype: np.dtype
    distance_metric: DistanceMetric

    num_clusters: int
    cluster_sample_size: int

    pool_sizes: PoolSizes
    segment_locality: SegmentLocality
    streaming_config: StreamingConfig

    groundtruth_k: int
    groundtruth_device: ComputeDevice
    output_format: OutputFormat
    random_seed: int

    def __post_init__(self) -> None:
        # Constructed purely for its own validation side effect: this
        # raises immediately (no dataset or RNG needed) if num_delete's
        # per-step schedule could ever ask to delete more rows than could
        # possibly be active at that point -- see StreamingPlan.
        StreamingPlan(self.streaming_config, self.pool_sizes)


class StreamingWorkloadOrganizer:
    """Runs the insert(+delete)+search streaming pipeline: cluster-labels
    the source dataset, splits it into a base/insert/query 3-pool split,
    then writes the insert pool and the query (search) pool out as an
    ordered sequence of per-step segment files (streaming.StreamingPlan)
    -- one insert file and one search-query file per step, sized
    independently (the insert:search rate ratio). One groundtruth
    checkpoint is computed per `StreamingConfig.checkpoint_steps` step --
    each scored against whichever rows are actually active at that point
    (streaming.compute_checkpoint_groundtruth) and against the same
    externally-provided evaluation query set at every checkpoint (see
    run()'s `eval_query_dataset_path`), since a single static groundtruth
    file would only be correct for the run's final step.

    `StreamingConfig.workload_kind` selects which streaming shape this
    is: WorkloadKind.INSERT_SEARCH (pure growth, `num_delete` must be 0)
    or WorkloadKind.INSERT_DELETE_SEARCH (each step also deletes its
    share of `num_delete` rows, chosen uniformly at random from whatever
    is active immediately before that step's own insert -- see
    streaming.py's module docstring). Both are run through this same
    class -- only the StreamingConfig differs. Unlike insert/search
    (a pure function of step number), the delete schedule is inherently
    sequential: run() itself owns the RNG and the running "active ids"
    set (base rows and every not-yet-deleted inserted row), updating it
    step by step as rows are inserted and randomly deleted.

    `StreamingOrganizerConfig.segment_locality` (streaming.SegmentLocality)
    is an orthogonal axis: it picks whether the insert stream's and
    search stream's per-step segments are cluster-aligned (ALIGN),
    insert-only cluster-ordered (NONALIGN), or neither (RANDOM) -- see
    SegmentLocality's docstring.

    Output layout (see StreamingWorkloadManifest for the exact per-step
    naming convention): `base_pool.*` and `eval_query_pool.*` are single
    files (never segmented -- base is a one-time bulk build, eval queries
    are the same fixed set graded at every checkpoint); `insert/`,
    `search_query/`, `delete/`, and `groundtruth/` are each a directory
    of one file per step (delete/groundtruth only where applicable).
    """

    def __init__(self, config: StreamingOrganizerConfig) -> None:
        self._config: StreamingOrganizerConfig = config

    def run(
        self,
        source_dataset_path: Path,
        eval_query_dataset_path: Path,
        output_root: Path,
        set_name: str | None = None,
    ) -> StreamingWorkloadManifest:
        """Runs one streaming workload generation, writing its output
        under `output_root/<set_name>/` (auto-numbered set_1, set_2, ...
        if `set_name` is not given). `source_dataset_path` is the base
        dataset the base/insert/query pools are cut from (query here
        means "vectors held out from insert/base, available to submit as
        stream traffic" -- see python/arachne/README.md).
        `eval_query_dataset_path` is a separate, already-final query set
        used purely to grade ground truth, mirroring how real datasets
        ship base + query as two distinct files."""
        resolved_set_name, output_dir = _resolve_output_dir(output_root, set_name)
        run_seed = _derive_run_seed(self._config.random_seed, resolved_set_name)
        streaming_config = self._config.streaming_config
        # Cheap, dataset-free validation (see StreamingPlan) -- catches an
        # infeasible delete schedule before opening anything.
        plan = StreamingPlan(streaming_config, self._config.pool_sizes)
        logger.info(
            "=== streaming workload generation start: set=%s workload_kind=%s segment_locality=%s "
            "num_steps=%d num_insert=%d num_search=%d num_delete=%d output_dir=%s run_seed=%d ===",
            resolved_set_name, streaming_config.workload_kind.value,
            self._config.segment_locality.value, streaming_config.num_steps,
            self._config.pool_sizes.num_insert, self._config.pool_sizes.num_query,
            streaming_config.num_delete, output_dir, run_seed,
        )

        try:
            with (
                open_source_reader(
                    source_dataset_path, self._config.source_format, dtype=self._config.dtype
                ) as reader,
                open_source_reader(
                    eval_query_dataset_path, self._config.source_format, dtype=self._config.dtype
                ) as eval_reader,
            ):
                if reader.dim != self._config.dim:
                    raise ValueError(
                        f"config dim={self._config.dim} does not match source "
                        f"dataset dim={reader.dim}"
                    )
                if eval_reader.dim != self._config.dim:
                    raise ValueError(
                        f"config dim={self._config.dim} does not match eval query "
                        f"dataset dim={eval_reader.dim}"
                    )
                # Cheap (header-only) check, before the expensive
                # clustering step below: fail fast if the requested pools
                # can't possibly fit in the source dataset.
                if self._config.pool_sizes.total > reader.num_vectors:
                    raise ValueError(
                        f"requested pool sizes sum to {self._config.pool_sizes.total}, which "
                        f"exceeds the source dataset size {reader.num_vectors}"
                    )

                assignment = _assign_clusters(
                    reader, self._config.num_clusters, self._config.cluster_sample_size, run_seed
                )
                # The insert stream's and search stream's ordering are
                # resolved together from segment_locality (see its
                # docstring): ALIGN keeps both cluster-ordered (so step i's
                # insert and search segments tend to share a cluster),
                # NONALIGN keeps only the insert stream cluster-ordered,
                # RANDOM scatters both.
                split = _split_pools(
                    assignment,
                    self._config.pool_sizes,
                    self._config.segment_locality.insert_order,
                    run_seed,
                    query_order=self._config.segment_locality.search_order,
                )

                output_format = self._config.output_format
                base_pool_path = output_dir / f"base_pool{output_format.pool_file_suffix(reader.dtype)}"
                insert_dir = output_dir / "insert"
                search_dir = output_dir / "search_query"
                delete_dir = output_dir / "delete"
                groundtruth_dir = output_dir / "groundtruth"
                for directory in (insert_dir, search_dir, delete_dir, groundtruth_dir):
                    directory.mkdir(parents=True, exist_ok=True)

                logger.info("writing base pool (%d rows) -> %s", split.base_rows.shape[0], base_pool_path)
                _write_pool(reader, split.base_rows, base_pool_path, output_format)

                eval_query_pool_path, eval_query_vectors = _write_eval_query_pool(
                    eval_reader, output_dir, output_format
                )

                computer = GroundTruthComputer(
                    k=self._config.groundtruth_k,
                    metric=self._config.distance_metric,
                    device=self._config.groundtruth_device,
                )

                # active_ids is the running set of currently-live global
                # ids: 0..num_base-1 is a position in split.base_rows,
                # num_base+j is position j in split.insert_rows (see
                # streaming.compute_checkpoint_groundtruth). Deletion picks
                # uniformly at random from this set as it stands *before*
                # that step's own insert is folded in (base rows are as
                # eligible as any already-inserted row); insert/search
                # ranges themselves are deterministic (StreamingPlan).
                num_base = split.base_rows.shape[0]
                active_ids: set[int] = set(range(num_base))
                rng = np.random.default_rng(run_seed)

                logger.info(
                    "writing %d step(s) of insert/search-query (+delete where applicable) -> %s, %s, %s",
                    streaming_config.num_steps, insert_dir, search_dir, delete_dir,
                )
                checkpoint_steps = streaming_config.checkpoint_steps
                for step in range(1, streaming_config.num_steps + 1):
                    delete_count = plan.delete_count(step)
                    if delete_count > 0:
                        candidates = np.array(sorted(active_ids), dtype=np.int64)
                        delete_ids = rng.choice(candidates, size=delete_count, replace=False)
                        active_ids.difference_update(delete_ids.tolist())
                        write_id_list(
                            step_file_path(delete_dir, step, output_format.id_list_file_suffix),
                            np.sort(delete_ids),
                            output_format,
                        )

                    insert_start, insert_end = plan.insert_range(step)
                    _write_pool(
                        reader,
                        split.insert_rows[insert_start:insert_end],
                        step_file_path(insert_dir, step, output_format.pool_file_suffix(reader.dtype)),
                        output_format,
                    )
                    active_ids.update(range(num_base + insert_start, num_base + insert_end))

                    search_start, search_end = plan.search_range(step)
                    _write_pool(
                        reader,
                        split.query_rows[search_start:search_end],
                        step_file_path(search_dir, step, output_format.pool_file_suffix(reader.dtype)),
                        output_format,
                    )
                    logger.debug(
                        "wrote step %d/%d (insert/search/delete)", step, streaming_config.num_steps
                    )

                    if step in checkpoint_steps:
                        gt_path = step_file_path(
                            groundtruth_dir, step, output_format.groundtruth_file_suffix
                        )
                        logger.info(
                            "computing checkpoint groundtruth (step=%d) -> %s", step, gt_path,
                        )
                        active_array = np.array(sorted(active_ids), dtype=np.int64)
                        result = compute_checkpoint_groundtruth(
                            computer, reader, split, active_array, eval_query_vectors
                        )
                        write_groundtruth(
                            gt_path, result.neighbor_ids, result.neighbor_dists, output_format
                        )

            manifest = self._write_manifest(
                set_name=resolved_set_name,
                source_dataset_path=source_dataset_path,
                eval_query_dataset_path=eval_query_dataset_path,
                output_dir=output_dir,
                split=split,
                base_pool_path=base_pool_path,
                insert_dir=insert_dir,
                search_dir=search_dir,
                delete_dir=delete_dir,
                groundtruth_dir=groundtruth_dir,
                eval_query_pool_path=eval_query_pool_path,
                eval_query_pool_count=int(eval_query_vectors.shape[0]),
                checkpoint_steps=checkpoint_steps,
                run_seed=run_seed,
            )
        except Exception:
            logger.error(
                "streaming workload generation failed for set=%s, removing %s",
                resolved_set_name, output_dir,
            )
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

        logger.info("=== streaming workload generation complete: set=%s ===", resolved_set_name)
        return manifest

    def _write_manifest(
        self,
        set_name: str,
        source_dataset_path: Path,
        eval_query_dataset_path: Path,
        output_dir: Path,
        split: PoolSplit,
        base_pool_path: Path,
        insert_dir: Path,
        search_dir: Path,
        delete_dir: Path,
        groundtruth_dir: Path,
        eval_query_pool_path: Path,
        eval_query_pool_count: int,
        checkpoint_steps: tuple[int, ...],
        run_seed: int,
    ) -> StreamingWorkloadManifest:
        streaming_config = self._config.streaming_config
        manifest = StreamingWorkloadManifest(
            set_name=set_name,
            source_dataset_path=str(source_dataset_path),
            dim=self._config.dim,
            dtype=np.dtype(self._config.dtype).name,
            num_clusters=self._config.num_clusters,
            output_format=self._config.output_format.value,
            base_pool_path=str(base_pool_path),
            base_pool_count=int(split.base_rows.shape[0]),
            segment_locality=self._config.segment_locality.value,
            insert_dir=str(insert_dir),
            insert_pool_count=int(split.insert_rows.shape[0]),
            insert_order=self._config.segment_locality.insert_order.value,
            insert_cluster_ranges=[
                ClusterRangeInfo(cluster_id=r.cluster_id, start=r.start, end=r.end)
                for r in split.insert_cluster_ranges
            ],
            search_dir=str(search_dir),
            search_pool_count=int(split.query_rows.shape[0]),
            search_order=self._config.segment_locality.search_order.value,
            search_cluster_ranges=[
                ClusterRangeInfo(cluster_id=r.cluster_id, start=r.start, end=r.end)
                for r in split.query_cluster_ranges
            ],
            delete_dir=str(delete_dir),
            eval_query_dataset_path=str(eval_query_dataset_path),
            eval_query_pool_path=str(eval_query_pool_path),
            eval_query_pool_count=eval_query_pool_count,
            workload_kind=streaming_config.workload_kind.value,
            num_steps=streaming_config.num_steps,
            num_delete=streaming_config.num_delete,
            checkpoint_every=streaming_config.checkpoint_every,
            checkpoint_steps=checkpoint_steps,
            groundtruth_dir=str(groundtruth_dir),
            groundtruth_k=self._config.groundtruth_k,
            random_seed=run_seed,
        )
        manifest_path = output_dir / "manifest.npz"
        manifest.to_npz(manifest_path)
        logger.info("wrote manifest -> %s", manifest_path)
        return manifest
