"""arachne.workload.pool_split

Splits a cluster-labeled base dataset into three non-overlapping pools
(base / insert / query):

  - base pool:   initial index build set.
  - insert pool: streaming-insert candidates. Row order within this pool
                 controls insert locality (see PoolRowOrder): a benchmark's
                 loader that only supports contiguous [start, end) reads
                 needs "insert cluster C only" to mean C's rows are
                 contiguous in the *output* file, not necessarily in the
                 original dataset.
  - query pool:  held out, never inserted. These are *stream queries* --
                 vectors available to submit as SEARCH traffic during a
                 benchmark run (e.g. to measure latency/throughput under
                 concurrent insert/delete) -- not the set ground truth is
                 graded against. Grading uses a separate, already-final
                 evaluation query set instead (see organizer.py's
                 `eval_query_dataset_path`), mirroring how real datasets
                 ship base and query as two distinct files. Row order
                 within this pool controls search locality the same way
                 PoolRowOrder controls insert locality (see split()'s
                 `query_order` -- streaming.SegmentLocality picks both
                 together).

Pool sizes are *exact counts* (PoolSizes), not ratios: the caller picks
precisely how many vectors go into each of the three pools (e.g. 1,000,000
base / 100,000 insert / 10,000 query), and the three counts need not sum to
the full dataset size -- any rows not claimed by a pool are simply left
unused. This matters because a source dataset can be far larger (e.g.
SIFT1B's 1e9 rows) than any single controlled experiment needs.

The split is still stratified per cluster: each pool's rows are drawn from
every cluster in proportion to that cluster's share of the *whole* dataset
(see _apportion_with_capacity), not just from a few large clusters, so no
pool silently over/under-represents a cluster relative to its natural
size.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from arachne.workload.clustering import ClusterAssignment
from arachne.workload.logging_utils import get_logger

logger = get_logger(__name__)


class PoolRowOrder(enum.Enum):
    """How rows within the insert pool are ordered in the output file.

    NATURAL: keep ascending original-dataset row order. This is the "고정된
        하나의 drift 패턴" (fixed drift pattern) baseline raised as a
        methodological concern; kept only as a regression reference, not
        as an O3 locality-control condition.
    CLUSTER: group rows by cluster id (ascending cluster id, ascending row
        id within a cluster). Enables contiguous cluster-only inserts via
        the existing start/end mechanism. Segments cut out of a
        CLUSTER-ordered pool each draw from (mostly) one cluster, and
        consecutive segments drift through clusters in ascending order.
    CLUSTER_SHUFFLED: same per-cluster contiguous grouping as CLUSTER (so
        segment-level locality/drift is preserved the same way), but the
        *sequence* clusters appear in is an independent random permutation
        instead of ascending cluster id. Used to make one stream (e.g. the
        search stream) still drift through one cluster at a time, without
        that drift being synchronized to another stream's (e.g. insert's)
        cluster sequence -- see streaming.SegmentLocality.NONALIGN.
    RANDOM: uniformly shuffled with an explicit seed, ignoring cluster
        structure entirely (every position is an equally likely draw from
        any cluster). Used as a locality-free control condition.
    """

    NATURAL = "natural"
    CLUSTER = "cluster"
    CLUSTER_SHUFFLED = "cluster_shuffled"
    RANDOM = "random"


@dataclass(frozen=True)
class PoolSizes:
    """Exact, absolute row counts for each pool -- not ratios.

    `num_base + num_insert + num_query` need not equal the dataset size;
    any rows not allocated to a pool are simply left unused. PoolSplitter
    raises if the requested total exceeds the dataset size.
    """

    num_base: int
    num_insert: int
    num_query: int

    def __post_init__(self) -> None:
        if self.num_base < 0 or self.num_insert < 0 or self.num_query < 0:
            raise ValueError(
                f"pool sizes must be non-negative, got "
                f"base={self.num_base} insert={self.num_insert} query={self.num_query}"
            )

    @property
    def total(self) -> int:
        return self.num_base + self.num_insert + self.num_query


@dataclass(frozen=True)
class ClusterRange:
    """Contiguous [start, end) row range of one cluster within a pool's
    *output* file -- directly usable as a svfusion runbook insert
    start/end, or as a query-file cluster-selection offset."""

    cluster_id: int
    start: int
    end: int


@dataclass(frozen=True)
class PoolSplit:
    """Result of splitting one cluster-labeled dataset into three pools.

    Each `*_rows` array holds row indices into the *original* dataset
    file, already ordered exactly as they should be written to that
    pool's output file (see organizer.py). `base_rows` is always sorted
    ascending, so a row's position in `base_rows` equals its row index in
    the written base-pool file (this is the contract groundtruth.py
    relies on). `insert_cluster_ranges` is only non-empty when the insert
    pool was built with PoolRowOrder.CLUSTER or CLUSTER_SHUFFLED (both
    preserve contiguous per-cluster blocks; NATURAL/RANDOM interleave
    clusters, so there is no single range per cluster to report).
    """

    base_rows: np.ndarray
    insert_rows: np.ndarray
    query_rows: np.ndarray
    insert_cluster_ranges: list[ClusterRange]
    query_cluster_ranges: list[ClusterRange]


def _apportion_with_capacity(weights: np.ndarray, total: int, capacities: np.ndarray) -> np.ndarray:
    """Distributes an integer `total` across len(weights) buckets, as
    proportionally to `weights` as integers allow, without exceeding each
    bucket's `capacities[i]`.

    Uses the largest-remainder method (compute each bucket's ideal
    fractional share, take the floor, then hand out the few leftover
    units to the buckets with the largest fractional remainder first) so
    the result sums to exactly `total` -- required here because the
    caller wants an *exact* pool size, not an approximate one. Any
    shortfall created by a bucket hitting its capacity cap is
    redistributed to buckets that still have spare capacity.

    Raises if sum(capacities) < total (the request is infeasible).
    """
    if total > int(capacities.sum()):
        raise ValueError(
            f"requested total {total} exceeds available capacity {int(capacities.sum())}"
        )
    counts = np.zeros_like(capacities)
    if total == 0:
        return counts

    normalized_weights = weights / weights.sum()
    ideal = normalized_weights * total
    counts = np.minimum(np.floor(ideal).astype(capacities.dtype), capacities)
    remainder = total - int(counts.sum())

    spare = capacities - counts
    if remainder > 0:
        # Largest fractional remainder first -- the classic largest-remainder
        # (Hamilton/Hare) apportionment method.
        fractional_part = ideal - np.floor(ideal)
        priority_order = np.argsort(-fractional_part)
        # One full pass hands one extra unit to each bucket with spare
        # capacity, in priority order; repeat until the remainder is used up.
        # This always terminates because sum(spare) >= remainder is
        # guaranteed by the capacity check above.
        while remainder > 0:
            progressed_this_pass = False
            for i in priority_order:
                if remainder == 0:
                    break
                if spare[i] > 0:
                    counts[i] += 1
                    spare[i] -= 1
                    remainder -= 1
                    progressed_this_pass = True
            if not progressed_this_pass:
                break  # defensive: should be unreachable given the check above
    return counts


class PoolSplitter:
    """Partitions cluster-labeled row indices into disjoint base/insert/
    query pools of exact requested sizes (PoolSizes), stratified by each
    cluster's natural share of the dataset, then orders the insert pool
    and the query pool independently, each per its own PoolRowOrder (see
    split()'s `insert_order`/`query_order`)."""

    def __init__(self, pool_sizes: PoolSizes, random_seed: int) -> None:
        self._pool_sizes: PoolSizes = pool_sizes
        self._random_seed: int = random_seed

    def split(
        self,
        assignment: ClusterAssignment,
        insert_order: PoolRowOrder,
        query_order: PoolRowOrder,
    ) -> PoolSplit:
        """`query_order` controls how the query pool is ordered, exactly
        like `insert_order` does for the insert pool -- the query pool is
        consumed as an ordered stream of per-step search segments (see
        StreamingWorkloadOrganizer), so its ordering matters the same way
        the insert pool's does. `streaming.SegmentLocality` picks
        `insert_order` and `query_order` together, since only certain
        combinations of the two are meaningful (see its docstring)."""
        rng = np.random.default_rng(self._random_seed)
        num_clusters = assignment.num_clusters

        # Shuffle each cluster's row indices once up front -- every
        # downstream slice (base/insert/query) then draws a uniformly
        # random subset of that cluster simply by taking a prefix.
        cluster_rows_shuffled: list[np.ndarray] = []
        cluster_sizes = np.zeros(num_clusters, dtype=np.int64)
        for cluster_id in range(num_clusters):
            rows = np.flatnonzero(assignment.cluster_ids == cluster_id)
            rng.shuffle(rows)
            cluster_rows_shuffled.append(rows)
            cluster_sizes[cluster_id] = rows.shape[0]

        dataset_size = int(cluster_sizes.sum())
        if self._pool_sizes.total > dataset_size:
            raise ValueError(
                f"requested pool sizes sum to {self._pool_sizes.total}, which "
                f"exceeds the dataset size {dataset_size}"
            )
        logger.info(
            "splitting %d rows across %d clusters into base=%d insert=%d query=%d "
            "(unused=%d)",
            dataset_size, num_clusters,
            self._pool_sizes.num_base, self._pool_sizes.num_insert, self._pool_sizes.num_query,
            dataset_size - self._pool_sizes.total,
        )

        # Apportion base, then insert, then query -- each call shrinks the
        # remaining per-cluster capacity so the three pools never overlap.
        # Weighting by the *original* cluster_sizes (not the shrinking
        # remaining capacity) keeps every pool's cluster composition
        # proportional to the dataset's natural cluster distribution.
        remaining_capacity = cluster_sizes.copy()
        base_counts = _apportion_with_capacity(
            cluster_sizes.astype(np.float64), self._pool_sizes.num_base, remaining_capacity
        )
        remaining_capacity -= base_counts
        insert_counts = _apportion_with_capacity(
            cluster_sizes.astype(np.float64), self._pool_sizes.num_insert, remaining_capacity
        )
        remaining_capacity -= insert_counts
        query_counts = _apportion_with_capacity(
            cluster_sizes.astype(np.float64), self._pool_sizes.num_query, remaining_capacity
        )

        base_rows_parts: list[np.ndarray] = []
        insert_rows_by_cluster: dict[int, np.ndarray] = {}
        query_rows_by_cluster: dict[int, np.ndarray] = {}
        for cluster_id in range(num_clusters):
            rows = cluster_rows_shuffled[cluster_id]
            n_base = int(base_counts[cluster_id])
            n_insert = int(insert_counts[cluster_id])
            n_query = int(query_counts[cluster_id])
            base_rows_parts.append(rows[:n_base])
            insert_rows_by_cluster[cluster_id] = rows[n_base : n_base + n_insert]
            query_rows_by_cluster[cluster_id] = rows[n_base + n_insert : n_base + n_insert + n_query]
            # rows[n_base + n_insert + n_query:] are simply left unused.
            logger.debug(
                "cluster %d: size=%d base=%d insert=%d query=%d unused=%d",
                cluster_id, rows.shape[0], n_base, n_insert, n_query,
                rows.shape[0] - n_base - n_insert - n_query,
            )

        base_rows = (
            np.sort(np.concatenate(base_rows_parts))
            if base_rows_parts
            else np.array([], dtype=np.int64)
        )
        insert_rows, insert_cluster_ranges = self._order_insert_pool(
            insert_rows_by_cluster, insert_order, rng
        )
        query_rows, query_cluster_ranges = self._order_insert_pool(
            query_rows_by_cluster, query_order, rng
        )

        return PoolSplit(
            base_rows=base_rows,
            insert_rows=insert_rows,
            query_rows=query_rows,
            insert_cluster_ranges=insert_cluster_ranges,
            query_cluster_ranges=query_cluster_ranges,
        )

    @staticmethod
    def _group_by_cluster_order(
        rows_by_cluster: dict[int, np.ndarray],
        cluster_order: list[int],
    ) -> tuple[np.ndarray, list[ClusterRange]]:
        """Concatenate rows in the given cluster-id order (ascending row id
        within each cluster), recording each cluster's [start, end) range
        in the concatenated output. Shared by CLUSTER (cluster_order =
        ascending cluster ids) and CLUSTER_SHUFFLED (cluster_order = a
        random permutation of them) -- both preserve per-cluster
        contiguous blocks, differing only in which order the blocks
        appear in."""
        ordered_parts: list[np.ndarray] = []
        ranges: list[ClusterRange] = []
        cursor = 0
        for cluster_id in cluster_order:
            part = np.sort(rows_by_cluster[cluster_id])
            ordered_parts.append(part)
            ranges.append(ClusterRange(cluster_id=cluster_id, start=cursor, end=cursor + part.shape[0]))
            cursor += part.shape[0]
        rows = np.concatenate(ordered_parts) if ordered_parts else np.array([], dtype=np.int64)
        return rows, ranges

    @staticmethod
    def _group_by_cluster(
        rows_by_cluster: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, list[ClusterRange]]:
        """Concatenate rows in ascending cluster-id order -- see
        _group_by_cluster_order."""
        return PoolSplitter._group_by_cluster_order(rows_by_cluster, sorted(rows_by_cluster.keys()))

    @staticmethod
    def _order_insert_pool(
        rows_by_cluster: dict[int, np.ndarray],
        insert_order: PoolRowOrder,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[ClusterRange]]:
        if insert_order is PoolRowOrder.CLUSTER:
            return PoolSplitter._group_by_cluster(rows_by_cluster)
        if insert_order is PoolRowOrder.CLUSTER_SHUFFLED:
            shuffled_cluster_order = list(rows_by_cluster.keys())
            rng.shuffle(shuffled_cluster_order)
            return PoolSplitter._group_by_cluster_order(rows_by_cluster, shuffled_cluster_order)

        all_rows = (
            np.concatenate(list(rows_by_cluster.values()))
            if rows_by_cluster
            else np.array([], dtype=np.int64)
        )
        if insert_order is PoolRowOrder.NATURAL:
            ordered_rows = np.sort(all_rows)
        elif insert_order is PoolRowOrder.RANDOM:
            ordered_rows = all_rows.copy()
            rng.shuffle(ordered_rows)
        else:
            raise ValueError(f"unhandled PoolRowOrder: {insert_order}")
        # NATURAL/RANDOM orders interleave clusters, so there is no single
        # contiguous [start, end) range per cluster to report.
        return ordered_rows, []
