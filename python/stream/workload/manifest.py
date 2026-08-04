"""arachne.workload.manifest

Serializable description of one organizer run's output: where each pool's
file lives, in which OutputFormat, and (for the insert/query pools) the
contiguous [start, end) row range of each cluster within that file. Those
ranges are what a benchmark run passes directly as an insert start/end,
or uses to select a per-cluster query file slice.

Deliberately NOT JSON: this is serialized as a numpy .npz archive (a zip
container of typed arrays) so the manifest stays in the same
binary/numpy-native family as the pool/groundtruth files it describes,
with no text/JSON parsing involved anywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClusterRangeInfo:
    cluster_id: int
    start: int
    end: int


@dataclass(frozen=True)
class StreamingWorkloadManifest:
    """Serializable description of one StreamingWorkloadOrganizer run.

    The insert pool, search (stream query) pool, delete-id lists, and
    per-checkpoint groundtruth are each a *directory* of one file per
    step, not a single file -- see
    organizer.py's `step_file_path()` for the shared naming convention
    (`step_{i:05d}<suffix>`, 1-indexed). Concretely, for step `i`:

      - `insert_dir/step_{i:05d}<pool_file_suffix>` -- always present;
        `num_insert` is divided into `num_steps` near-equal per-step
        counts (see streaming.StreamingPlan), so a step's row count can
        differ by at most 1 from any other's -- read the file itself
        (its own xbin/npy header) for the exact count. `pool_file_suffix`
        (OutputFormat.pool_file_suffix(dtype)) is dtype-specific for
        OutputFormat.XBIN (e.g. ".u8bin" for uint8, ".fbin" for float32,
        matching big-ann-benchmarks/cuVS's own naming convention) -- this
        manifest's own `dtype` field plus `output_format` is enough for a
        consumer to reconstruct it without re-deriving it another way.
      - `search_dir/step_{i:05d}<pool_file_suffix>` -- likewise, always
        present, `num_query`'s share of that step.
      - `delete_dir/step_{i:05d}<id_list_file_suffix>` -- present only if
        this step's share of `num_delete` is non-zero; a plain, sorted
        list of the *global* ids being deleted (0..base_pool_count-1 is a
        position in the base pool, base_pool_count+j is insert-stream
        position j) -- chosen uniformly at random from whatever was
        active immediately before this step's own insert, not
        necessarily a contiguous range (see streaming.py). Ids only, not
        vector data (the consumer already has those from when they were
        inserted, or from the base pool build).
      - `groundtruth_dir/step_{i:05d}<groundtruth_file_suffix>` --
        present only at checkpoint steps (`i` in `checkpoint_steps`).

    `insert_cluster_ranges`/`search_cluster_ranges` describe [start, end)
    ranges in the *global* position space across the whole insert/search
    stream (0..insert_pool_count / 0..search_pool_count) -- unlike step
    boundaries (which vary by at most 1 row and so aren't recoverable by
    simple division anymore), these ranges are recorded directly and
    don't depend on the per-step split at all.
    """

    set_name: str

    source_dataset_path: str
    dim: int
    dtype: str
    num_clusters: int
    output_format: str

    base_pool_path: str
    base_pool_count: int

    # streaming.SegmentLocality.value: which of the three (insert_order,
    # search_order) combinations below this run used -- see
    # StreamingOrganizerConfig.segment_locality. insert_order/search_order
    # are the concrete, resolved PoolRowOrder values it implies, kept
    # alongside for any consumer that just wants to know how to interpret
    # insert_cluster_ranges/search_cluster_ranges without needing the
    # SegmentLocality -> (insert_order, search_order) mapping itself.
    segment_locality: str

    insert_dir: str
    insert_pool_count: int  # total rows across every step file
    insert_order: str
    insert_cluster_ranges: list[ClusterRangeInfo]  # global positions, see class docstring

    search_dir: str  # "stream queries": SEARCH traffic issued alongside each insert step
    search_pool_count: int  # total rows across every step file
    search_order: str
    search_cluster_ranges: list[ClusterRangeInfo]

    delete_dir: str  # per-step node-id lists; empty dir if nothing is ever deleted

    # Separate, already-final query set used purely to grade ground truth
    # at every checkpoint -- see organizer.py.
    eval_query_dataset_path: str
    eval_query_pool_path: str
    eval_query_pool_count: int

    workload_kind: str  # streaming.WorkloadKind.value
    num_steps: int
    num_delete: int  # 0 for WorkloadKind.INSERT_SEARCH
    checkpoint_every: int
    # Plain tuple (not np.ndarray): dataclass equality compares fields via
    # tuple(...) == tuple(...), and a multi-element ndarray there raises
    # "truth value is ambiguous" instead of comparing -- a tuple avoids that.
    checkpoint_steps: tuple[int, ...]
    groundtruth_dir: str
    groundtruth_k: int

    random_seed: int

    def to_npz(self, path: Path) -> None:
        np.savez(
            path,
            set_name=np.array(self.set_name),
            source_dataset_path=np.array(self.source_dataset_path),
            dim=np.array(self.dim),
            dtype=np.array(self.dtype),
            num_clusters=np.array(self.num_clusters),
            output_format=np.array(self.output_format),
            base_pool_path=np.array(self.base_pool_path),
            base_pool_count=np.array(self.base_pool_count),
            segment_locality=np.array(self.segment_locality),
            insert_dir=np.array(self.insert_dir),
            insert_pool_count=np.array(self.insert_pool_count),
            insert_order=np.array(self.insert_order),
            insert_cluster_ranges=_ranges_to_array(self.insert_cluster_ranges),
            search_dir=np.array(self.search_dir),
            search_pool_count=np.array(self.search_pool_count),
            search_order=np.array(self.search_order),
            search_cluster_ranges=_ranges_to_array(self.search_cluster_ranges),
            delete_dir=np.array(self.delete_dir),
            eval_query_dataset_path=np.array(self.eval_query_dataset_path),
            eval_query_pool_path=np.array(self.eval_query_pool_path),
            eval_query_pool_count=np.array(self.eval_query_pool_count),
            workload_kind=np.array(self.workload_kind),
            num_steps=np.array(self.num_steps),
            num_delete=np.array(self.num_delete),
            checkpoint_every=np.array(self.checkpoint_every),
            checkpoint_steps=np.array(self.checkpoint_steps, dtype=np.int64),
            groundtruth_dir=np.array(self.groundtruth_dir),
            groundtruth_k=np.array(self.groundtruth_k),
            random_seed=np.array(self.random_seed),
        )

    @staticmethod
    def from_npz(path: Path) -> "StreamingWorkloadManifest":
        with np.load(path) as data:
            return StreamingWorkloadManifest(
                set_name=str(data["set_name"]),
                source_dataset_path=str(data["source_dataset_path"]),
                dim=int(data["dim"]),
                dtype=str(data["dtype"]),
                num_clusters=int(data["num_clusters"]),
                output_format=str(data["output_format"]),
                base_pool_path=str(data["base_pool_path"]),
                base_pool_count=int(data["base_pool_count"]),
                segment_locality=str(data["segment_locality"]),
                insert_dir=str(data["insert_dir"]),
                insert_pool_count=int(data["insert_pool_count"]),
                insert_order=str(data["insert_order"]),
                insert_cluster_ranges=_array_to_ranges(data["insert_cluster_ranges"]),
                search_dir=str(data["search_dir"]),
                search_pool_count=int(data["search_pool_count"]),
                search_order=str(data["search_order"]),
                search_cluster_ranges=_array_to_ranges(data["search_cluster_ranges"]),
                delete_dir=str(data["delete_dir"]),
                eval_query_dataset_path=str(data["eval_query_dataset_path"]),
                eval_query_pool_path=str(data["eval_query_pool_path"]),
                eval_query_pool_count=int(data["eval_query_pool_count"]),
                workload_kind=str(data["workload_kind"]),
                num_steps=int(data["num_steps"]),
                num_delete=int(data["num_delete"]),
                checkpoint_every=int(data["checkpoint_every"]),
                checkpoint_steps=tuple(int(s) for s in data["checkpoint_steps"]),
                groundtruth_dir=str(data["groundtruth_dir"]),
                groundtruth_k=int(data["groundtruth_k"]),
                random_seed=int(data["random_seed"]),
            )


def _ranges_to_array(ranges: list[ClusterRangeInfo]) -> np.ndarray:
    """Encodes a list of ClusterRangeInfo as an (N, 3) int64 array of
    (cluster_id, start, end) rows -- npz has no native list-of-struct type,
    so this is the array-native equivalent."""
    if not ranges:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array([[r.cluster_id, r.start, r.end] for r in ranges], dtype=np.int64)


def _array_to_ranges(array: np.ndarray) -> list[ClusterRangeInfo]:
    return [
        ClusterRangeInfo(cluster_id=int(row[0]), start=int(row[1]), end=int(row[2]))
        for row in array
    ]
