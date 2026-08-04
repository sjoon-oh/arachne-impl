"""arachne.workload.groundtruth

Exact (brute-force) k-NN ground truth computation for a query pool against
a base pool, per SVFusion.pdf sec 6.1 ("ground truth computed via an
exhaustive linear scan"). Supports the two metrics svfusion's
ffanns::DistanceType actually uses (datasets.hpp): Euclidean and inner
product.

The distance/top-k arithmetic can run on CPU (numpy, always available) or
GPU (ComputeDevice.GPU, via the optional `cupy` package -- already present
in this environment as a RAFT/RMM transitive dependency, and a natural fit
since arachne targets a CUDA/Blackwell GPU host anyway). Which device runs
the math is purely an internal implementation detail: GroundTruthResult
always holds plain numpy arrays regardless of `device`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from arachne.workload.formats import SourceVectorReader
from arachne.workload.logging_utils import get_logger

logger = get_logger(__name__)


class DistanceMetric(enum.Enum):
    EUCLIDEAN = "euclidean"
    INNER_PRODUCT = "inner_product"


class ComputeDevice(enum.Enum):
    """Where GroundTruthComputer runs its distance/top-k arithmetic.

    CPU: numpy. Always available, no extra dependency.
    GPU: cupy, a numpy-compatible GPU array library. Constructing a
        GroundTruthComputer with ComputeDevice.GPU raises immediately if
        cupy is not importable, rather than failing deep inside compute().
    """

    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class GroundTruthResult:
    # neighbor_ids[q, r] is a row index into the *base pool's output file*
    # (i.e. a position in the sorted base_rows array passed to compute()),
    # not a row index into the original source dataset.
    neighbor_ids: np.ndarray  # shape (num_queries, k), dtype int32
    neighbor_dists: np.ndarray  # shape (num_queries, k), dtype float32


class GroundTruthComputer:
    """Exhaustive linear-scan k-NN: for every query, compute its distance
    to every base vector and keep the top-k.

    This is O(num_queries * num_base) time -- only tractable because the
    base pool used by experiment-plan.md's E4 (a 3-pool split of one
    dataset, not the full 1B-row SIFT1B) is kept in the tens-of-millions
    range at most. Base vectors are streamed in batches so the full base
    pool is never materialized in memory (host or device) at once.
    """

    def __init__(
        self,
        k: int,
        metric: DistanceMetric,
        base_batch_size: int = 40_000,
        device: ComputeDevice = ComputeDevice.CPU,
    ) -> None:
        self._k: int = k
        self._metric: DistanceMetric = metric
        self._base_batch_size: int = base_batch_size
        self._device: ComputeDevice = device
        self._xp: Any = self._resolve_array_module(device)

    @staticmethod
    def _resolve_array_module(device: ComputeDevice) -> Any:
        if device is ComputeDevice.CPU:
            return np
        if device is ComputeDevice.GPU:
            try:
                import cupy as cp
            except ImportError as e:
                raise RuntimeError(
                    "ComputeDevice.GPU requires the optional 'cupy' package, which "
                    "is not importable in this environment. Install a cupy build "
                    "matching your CUDA toolkit (e.g. cupy-cuda12x), or construct "
                    "GroundTruthComputer with ComputeDevice.CPU instead."
                ) from e
            return cp
        raise ValueError(f"unhandled ComputeDevice: {device}")

    def compute(
        self,
        reader: SourceVectorReader,
        base_rows: np.ndarray,
        query_vectors: np.ndarray,
    ) -> GroundTruthResult:
        """`base_rows[i]` is the row `reader` should be read from to
        produce position `i` of the pool being scored -- neighbor_ids in
        the result are these positions (0-indexed into base_rows), not
        raw reader row ids. `base_rows` may be in *any* order: the top-k
        merge below tracks each candidate's position in `base_rows`
        directly (an arange over the batch), so there is no sorted-order
        requirement and no row-id -> position lookup at the end. This
        matters for streaming checkpoints (streaming.py), whose active
        set is a concatenation of the base pool (sorted) with an
        insert-pool segment range that is generally *not* sorted by
        original-dataset row id (e.g. PoolRowOrder.CLUSTER/RANDOM)."""
        if base_rows.shape[0] < self._k:
            raise ValueError(
                f"base pool has {base_rows.shape[0]} rows, fewer than k={self._k}"
            )

        xp = self._xp
        num_queries = query_vectors.shape[0]
        # xp.asarray both casts to float32 and, for the GPU backend, copies
        # host memory to device memory -- this is the only host->device
        # transfer of the (small) query set; base vectors stream in below.
        query_dev = xp.asarray(query_vectors, dtype=xp.float32)
        logger.info(
            "computing exhaustive groundtruth on %s: num_queries=%d num_base=%d k=%d metric=%s",
            self._device.value, num_queries, base_rows.shape[0], self._k, self._metric.value,
        )

        best_dists = xp.full((num_queries, self._k), xp.inf, dtype=xp.float32)
        best_ids = xp.full((num_queries, self._k), -1, dtype=xp.int64)

        num_batches = (base_rows.shape[0] + self._base_batch_size - 1) // self._base_batch_size
        for batch_index, batch_start in enumerate(range(0, base_rows.shape[0], self._base_batch_size)):
            batch_row_ids_host = base_rows[batch_start : batch_start + self._base_batch_size]
            batch_vectors_host = reader.read_rows(batch_row_ids_host).astype(np.float32)
            # Reading from the source file is always host-side (readers do
            # plain file I/O); only the per-batch array math below moves
            # to the device for ComputeDevice.GPU.
            batch_vectors_dev = xp.asarray(batch_vectors_host)
            batch_positions_dev = xp.arange(batch_start, batch_start + batch_row_ids_host.shape[0])

            batch_dists = self._pairwise_distance(xp, query_dev, batch_vectors_dev)

            combined_dists = xp.concatenate([best_dists, batch_dists], axis=1)
            combined_ids = xp.concatenate(
                [best_ids, xp.broadcast_to(batch_positions_dev, (num_queries, batch_positions_dev.shape[0]))],
                axis=1,
            )
            top_k_idx = xp.argpartition(combined_dists, self._k - 1, axis=1)[:, : self._k]
            best_dists = xp.take_along_axis(combined_dists, top_k_idx, axis=1)
            best_ids = xp.take_along_axis(combined_ids, top_k_idx, axis=1)
            logger.debug("groundtruth base batch %d/%d done", batch_index + 1, num_batches)

        # Final sort within the k slots so neighbor_ids[:, 0] is nearest.
        order = xp.argsort(best_dists, axis=1)
        best_dists = xp.take_along_axis(best_dists, order, axis=1)
        best_ids = xp.take_along_axis(best_ids, order, axis=1)

        # best_ids already holds positions within base_rows (= row index in
        # the pool's output file) -- see the docstring above.
        # GroundTruthResult always holds plain numpy arrays -- which device
        # did the arithmetic is not the caller's concern.
        neighbor_ids = self._to_numpy(best_ids).astype(np.int32)
        neighbor_dists = self._to_numpy(best_dists)
        logger.info("groundtruth computation complete for %d queries", num_queries)

        return GroundTruthResult(neighbor_ids=neighbor_ids, neighbor_dists=neighbor_dists)

    @staticmethod
    def _to_numpy(array: Any) -> np.ndarray:
        # cupy arrays expose .get() to copy device->host; plain numpy
        # arrays have no such method and are returned as-is.
        return array.get() if hasattr(array, "get") else array

    def _pairwise_distance(self, xp: Any, queries: Any, base_vectors: Any) -> Any:
        """Returns a (num_queries, num_base) matrix where smaller is
        always closer, regardless of metric (inner product is negated so
        both branches share the same "smaller is better" convention used
        by the top-k merge above). `xp` is either numpy or cupy -- the two
        libraries' array APIs are compatible enough that this code is
        identical for both."""
        if self._metric is DistanceMetric.EUCLIDEAN:
            # ||q - b||^2 expanded via the dot-product identity to avoid
            # materializing a (num_queries, num_base, dim) tensor.
            query_sq = xp.sum(queries**2, axis=1, keepdims=True)
            base_sq = xp.sum(base_vectors**2, axis=1, keepdims=True).T
            cross_term = queries @ base_vectors.T
            return query_sq + base_sq - 2.0 * cross_term
        elif self._metric is DistanceMetric.INNER_PRODUCT:
            return -(queries @ base_vectors.T)
        else:
            raise ValueError(f"unhandled DistanceMetric: {self._metric}")
