"""arachne.workload.clustering

K-means cluster labeling for the base dataset, mirroring the "Clustered"
workload definition in SVFusion.pdf sec 6.1 (64 k-means clusters over the
dataset, from the 2023 Big ANN Challenge). Cluster labels are the unit of
locality control used downstream by pool_split.py to build aligned/
divergent insert-vs-query conditions for experiment-plan.md's E4.

Fitting k-means on the full dataset is unnecessary and often infeasible at
10M+ row scale, so this fits on a random sample and then assigns every row
in a second, sequential (batched) pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from arachne.workload.formats import SourceVectorReader
from arachne.workload.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClusterAssignment:
    """cluster_ids[i] is the cluster label of base-dataset row i."""

    cluster_ids: np.ndarray  # shape (num_vectors,), dtype int32
    num_clusters: int
    centroids: np.ndarray  # shape (num_clusters, dim), dtype float32


class KMeansClusterAssigner:
    """Fits k-means on a sample of a dataset, then labels every row."""

    def __init__(
        self,
        num_clusters: int,
        sample_size: int,
        random_seed: int,
        assign_batch_size: int = 1_000_000,
    ) -> None:
        self._num_clusters: int = num_clusters
        self._sample_size: int = sample_size
        self._random_seed: int = random_seed
        self._assign_batch_size: int = assign_batch_size

    def fit_and_assign(self, reader: SourceVectorReader) -> ClusterAssignment:
        rng = np.random.default_rng(self._random_seed)

        sample_size = min(self._sample_size, reader.num_vectors)
        logger.info(
            "fitting k-means: k=%d sample_size=%d (of %d total rows)",
            self._num_clusters, sample_size, reader.num_vectors,
        )
        sample_rows = rng.choice(reader.num_vectors, size=sample_size, replace=False)
        sample_vectors = reader.read_rows(sample_rows).astype(np.float32)

        kmeans = KMeans(
            n_clusters=self._num_clusters,
            random_state=self._random_seed,
            n_init="auto",
        )
        kmeans.fit(sample_vectors)
        logger.info("k-means fit complete, assigning cluster labels to all %d rows", reader.num_vectors)

        cluster_ids = np.empty(reader.num_vectors, dtype=np.int32)
        for batch_start in range(0, reader.num_vectors, self._assign_batch_size):
            batch_count = min(self._assign_batch_size, reader.num_vectors - batch_start)
            batch_vectors = reader.read_range(batch_start, batch_count).astype(np.float32)
            cluster_ids[batch_start : batch_start + batch_count] = kmeans.predict(batch_vectors)
            logger.debug(
                "assigned cluster labels for rows [%d, %d) / %d",
                batch_start, batch_start + batch_count, reader.num_vectors,
            )

        cluster_sizes = np.bincount(cluster_ids, minlength=self._num_clusters)
        logger.info(
            "cluster assignment complete: sizes min=%d max=%d mean=%.1f",
            int(cluster_sizes.min()), int(cluster_sizes.max()), float(cluster_sizes.mean()),
        )

        return ClusterAssignment(
            cluster_ids=cluster_ids,
            num_clusters=self._num_clusters,
            centroids=kmeans.cluster_centers_.astype(np.float32),
        )
