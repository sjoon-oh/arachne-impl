"""arachne.workload.streaming

Turns an insert pool and a search (stream query) pool into a shared
sequence of discrete "steps" (segments) -- one insert segment, one search
segment, and (optionally) one random-delete batch per step. `num_insert`/
`num_query`/`num_delete` (PoolSizes / StreamingConfig) are each a *total*
across the whole run; `num_steps` divides each total into per-step counts
as evenly as an integer split allows (`_split_into_steps`) -- so the
insert:search rate is simply whatever ratio the two totals are chosen in,
with no separate per-step size knob to keep in sync. This is for
SVFusion.pdf sec 6.1's two insert(+delete)+search workload shapes:

  - WorkloadKind.INSERT_SEARCH: pure growth. Every step inserts one
    segment; nothing is ever deleted (`num_delete` must be 0).
  - WorkloadKind.INSERT_DELETE_SEARCH: every step inserts one segment and
    (per its share of `num_delete`) deletes that many vectors, chosen
    *uniformly at random* from whatever is active immediately before this
    step's own insert (base pool rows and every not-yet-deleted
    previously-inserted row alike -- there is no FIFO/oldest-first rule).

Both kinds interleave SEARCH at configurable checkpoint steps. Each
checkpoint needs its *own* ground truth: the active row set changes from
step to step (segments arrive, and for INSERT_DELETE_SEARCH, random rows
expire), so one groundtruth file computed once against the full insert
pool would be wrong for every step except the very last.
`compute_checkpoint_groundtruth` below produces one correct,
checkpoint-specific result for each checkpoint, reusing the same
GroundTruthComputer unchanged for every one of them.

Unlike the insert/search schedule (a pure function of step number, so
`StreamingPlan.insert_range`/`.search_range` are O(1) closed-form
arithmetic), *which* rows get deleted is not: it depends on the actual
random draw at every earlier step, so it can only be computed by actually
replaying the run step by step (see StreamingWorkloadOrganizer.run(),
which owns the RNG and the running "active ids" set). `StreamingPlan`
only knows *how many* to delete at a given step (`delete_count`), plus
validates up front, from `num_base`/`num_insert`/`num_delete` alone (no
RNG or dataset access needed), that the schedule can never ask a step to
delete more rows than could possibly be active at that point.

See organizer.py's StreamingWorkloadOrganizer for how this plugs into the
existing cluster-labeling / base-insert-query pool split, and
python/arachne/README.md for the step/checkpoint timeline diagram.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from arachne.workload.formats import SourceVectorReader
from arachne.workload.groundtruth import GroundTruthComputer, GroundTruthResult
from arachne.workload.pool_split import PoolRowOrder, PoolSizes, PoolSplit


class WorkloadKind(enum.Enum):
    """Which streaming workload this is -- see SVFusion.pdf sec 6.1.

    INSERT_SEARCH: every step inserts one segment; the active set only
        ever grows. Requires `StreamingConfig.num_delete == 0`.
    INSERT_DELETE_SEARCH: every step inserts one segment and deletes its
        share of `num_delete` vectors, chosen uniformly at random from
        whatever is active at that point (see module docstring).
    """

    INSERT_SEARCH = "insert_search"
    INSERT_DELETE_SEARCH = "insert_delete_search"


class SegmentLocality(enum.Enum):
    """How the insert stream's and search stream's per-step segments
    relate to each other -- the three combinations of (insert ordering,
    search ordering) that are actually meaningful to distinguish, out of
    the full PoolRowOrder x PoolRowOrder cross product (most of which
    collapse to one of these three anyway: NATURAL and RANDOM produce the
    same "no per-segment cluster concentration" outcome as far as this
    axis is concerned, and pairing a RANDOM/NATURAL insert stream with a
    CLUSTER-family search stream gives insert no "hot" cluster for search
    to align with or diverge from, so it isn't a useful combination
    either).

    ALIGN: both streams are ordered by ascending cluster id
        (PoolRowOrder.CLUSTER), so step i's insert segment and step i's
        search segment tend to draw from the same cluster region --
        high per-segment insert/search locality, with drift over time as
        later steps move on to later clusters.
    NONALIGN: the insert stream keeps its cluster-ordered drift
        (PoolRowOrder.CLUSTER), and the search stream *also* drifts
        through one cluster at a time (PoolRowOrder.CLUSTER_SHUFFLED --
        same per-segment cluster locality as CLUSTER, just a cluster
        sequence shuffled independently of insert's), so each individual
        search segment still has real intra-segment locality -- it's only
        the *correlation between* insert's and search's current cluster,
        segment by segment, that's removed.
    RANDOM: both streams are shuffled uniformly (PoolRowOrder.RANDOM) --
        neither drifts, and neither has any intra-segment cluster
        locality either; a flat baseline.
    """

    ALIGN = "align"
    NONALIGN = "nonalign"
    RANDOM = "random"

    @property
    def insert_order(self) -> PoolRowOrder:
        return PoolRowOrder.RANDOM if self is SegmentLocality.RANDOM else PoolRowOrder.CLUSTER

    @property
    def search_order(self) -> PoolRowOrder:
        if self is SegmentLocality.ALIGN:
            return PoolRowOrder.CLUSTER
        if self is SegmentLocality.NONALIGN:
            return PoolRowOrder.CLUSTER_SHUFFLED
        return PoolRowOrder.RANDOM


@dataclass(frozen=True)
class StreamingConfig:
    """Everything needed to divide one run into ordered steps, other than
    the pool sizes themselves (see StreamingPlan, which takes both this
    and a PoolSizes).

    `num_steps` is the "iteration set" size: how many discrete insert
    (+ maybe delete) steps this streaming run has (SVFusion's T_max).
    `PoolSizes.num_insert`/`num_query` (the insert pool's and search
    pool's *total* row counts) are each divided into `num_steps`
    near-equal per-step segments (`_split_into_steps`) -- so the
    insert:search rate is simply whatever ratio those two totals are
    chosen in, with no separate per-step size to keep in sync with them.

    `num_delete` is the *total* number of vectors deleted over the whole
    run (0 by default -- pure growth, WorkloadKind.INSERT_SEARCH). Like
    num_insert/num_query, it is divided into `num_steps` near-equal
    per-step counts; each step deletes that many rows, chosen uniformly
    at random from whatever is active immediately before that step's own
    insert (see module docstring -- there is no FIFO/oldest-first rule,
    and base-pool rows are as eligible for deletion as any inserted row).

    `checkpoint_every` controls how often (in steps) a SEARCH-with-
    groundtruth checkpoint runs: 1 means every step (SVFusion's own
    granularity), a larger value trades checkpoint resolution for less
    ground-truth recomputation cost. The final step is always a
    checkpoint, regardless of `checkpoint_every`, so a run's terminal
    state is never left unmeasured.
    """

    workload_kind: WorkloadKind
    num_steps: int
    num_delete: int = 0
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.num_delete < 0:
            raise ValueError(f"num_delete must be non-negative, got {self.num_delete}")
        if self.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be positive, got {self.checkpoint_every}")
        if self.workload_kind is WorkloadKind.INSERT_SEARCH and self.num_delete != 0:
            raise ValueError(
                f"num_delete must be 0 for WorkloadKind.INSERT_SEARCH, got {self.num_delete} "
                f"(use WorkloadKind.INSERT_DELETE_SEARCH for a non-zero delete count)"
            )

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        """Steps (1-indexed) at which a SEARCH+groundtruth checkpoint runs."""
        steps = [s for s in range(1, self.num_steps + 1) if s % self.checkpoint_every == 0]
        if not steps or steps[-1] != self.num_steps:
            steps.append(self.num_steps)
        return tuple(steps)


def _split_into_steps(total: int, num_steps: int) -> list[int]:
    """Divides `total` into `num_steps` non-negative integers summing
    exactly to `total`, as evenly as integer division allows -- the
    remainder (`total % num_steps`) is handed out one unit each to the
    first `total % num_steps` steps, so step sizes never differ by more
    than 1."""
    quotient, remainder = divmod(total, num_steps)
    return [quotient + (1 if i < remainder else 0) for i in range(num_steps)]


class StreamingPlan:
    """Derived view over a (StreamingConfig, PoolSizes) pair: for any
    step, how many insert-pool rows arrive, how many search-pool rows are
    issued, and how many rows get deleted.

    Insert/search are a pure function of step number (`insert_range`/
    `search_range` are O(1) closed-form arithmetic over precomputed
    per-step sizes), but *which* rows get deleted is not -- it depends on
    the actual random draw made at every earlier step, so only the
    *count* to delete at a given step (`delete_count`) is available here;
    the deletion itself is carried out sequentially by
    StreamingWorkloadOrganizer.run(), which owns the RNG and the running
    "active ids" set.

    What *is* checked here, eagerly and without touching the dataset or
    any RNG, is whether the schedule is even possible: cumulatively
    tracking how many rows would be active immediately before each step's
    delete (starting from `pool_sizes.num_base`, growing by that step's
    insert count and shrinking by its delete count), raising immediately
    if any step's delete_count would exceed that -- this is pure integer
    arithmetic over the two configs, so it fails fast, before any
    clustering or dataset I/O happens (see StreamingOrganizerConfig,
    which constructs one of these purely for this validation).
    """

    def __init__(self, config: StreamingConfig, pool_sizes: PoolSizes) -> None:
        self._config = config
        self._pool_sizes = pool_sizes

        self.insert_counts: list[int] = _split_into_steps(pool_sizes.num_insert, config.num_steps)
        self.search_counts: list[int] = _split_into_steps(pool_sizes.num_query, config.num_steps)
        self.delete_counts: list[int] = _split_into_steps(config.num_delete, config.num_steps)

        insert_cumulative = np.concatenate([[0], np.cumsum(self.insert_counts)])
        search_cumulative = np.concatenate([[0], np.cumsum(self.search_counts)])
        self._insert_cumulative: np.ndarray = insert_cumulative
        self._search_cumulative: np.ndarray = search_cumulative

        active = pool_sizes.num_base
        for step_idx in range(config.num_steps):
            delete_count = self.delete_counts[step_idx]
            if delete_count > active:
                raise ValueError(
                    f"step {step_idx + 1}: schedule requires deleting {delete_count} "
                    f"vectors, but only {active} would be active at that point "
                    f"(num_base={pool_sizes.num_base}, cumulative insert before this "
                    f"step={int(insert_cumulative[step_idx])}, cumulative delete before "
                    f"this step={int(np.sum(self.delete_counts[:step_idx]))}) -- lower "
                    f"num_delete, raise num_base/num_insert, or spread deletes over more "
                    f"num_steps"
                )
            active = active - delete_count + self.insert_counts[step_idx]

    def insert_range(self, step: int) -> tuple[int, int]:
        """[start, end) insert-pool row range inserted at `step`."""
        self._validate_step(step)
        return int(self._insert_cumulative[step - 1]), int(self._insert_cumulative[step])

    def search_range(self, step: int) -> tuple[int, int]:
        """[start, end) search-pool (stream query) row range issued at
        `step`, alongside that step's insert segment."""
        self._validate_step(step)
        return int(self._search_cumulative[step - 1]), int(self._search_cumulative[step])

    def delete_count(self, step: int) -> int:
        """How many rows get deleted at `step` (0 if nothing does)."""
        self._validate_step(step)
        return self.delete_counts[step - 1]

    def _validate_step(self, step: int) -> None:
        if not (1 <= step <= self._config.num_steps):
            raise ValueError(f"step {step} out of range [1, {self._config.num_steps}]")


def compute_checkpoint_groundtruth(
    computer: GroundTruthComputer,
    reader: SourceVectorReader,
    split: PoolSplit,
    active_ids: np.ndarray,
    query_vectors: np.ndarray,
) -> GroundTruthResult:
    """Ground truth for exactly the active set given by `active_ids`
    (sorted, deduplicated global ids: `0..num_base-1` is a position in
    `split.base_rows`, `num_base + j` is position `j` in
    `split.insert_rows` -- see StreamingWorkloadOrganizer.run(), which
    tracks this set across steps as rows are inserted and randomly
    deleted). Position `i` of the result corresponds to `active_ids[i]`,
    not to a fixed row range -- a downstream consumer that independently
    replays the exact same insert/delete files (and sorts its own active
    id set the same way) arrives at the identical `active_ids` array, and
    so can match GT positions to its own vectors without arachne needing
    to persist this array separately.

    Reads directly from the still-open source-dataset `reader` (via
    `split.base_rows`/`split.insert_rows`, which are row indices into
    that same source file) rather than reopening the written pool
    files -- one fewer file format to worry about, and avoids a
    redundant round trip through disk.
    """
    num_base = split.base_rows.shape[0]
    is_base = active_ids < num_base
    active_source_rows = np.empty_like(active_ids)
    active_source_rows[is_base] = split.base_rows[active_ids[is_base]]
    active_source_rows[~is_base] = split.insert_rows[active_ids[~is_base] - num_base]
    return computer.compute(reader, active_source_rows, query_vectors)
