"""arachne.workload

Dataset organizer for workload generation: splits a raw base dataset into
base/insert/query pools with cluster-controlled locality (k-means labels),
writes them back out in a pluggable OutputFormat (a simple fixed-header
xbin layout, or plain numpy), and computes exhaustive-scan ground truth
against a separate, externally-provided evaluation query set. Each run
produces one numbered "set" (set_1, set_2, ...) under an output root, so
repeated generations accumulate as distinct, comparable sets. See
python/arachne/README.md for the full pipeline diagram.

Two *source* dataset layouts are supported (SourceFormat, see formats.py):
SourceFormat.XBIN (the big-ann-benchmarks/DiskANN "xbin" convention --
also what Microsoft SPACEV1B uses, dtype=int8) and SourceFormat.VECS (the
classic INRIA/TEXMEX fvecs/ivecs/bvecs layout). See formats.py for how to
add another.

StreamingWorkloadOrganizer (organizer.py) runs an insert(+delete)+search
workload (streaming.py: WorkloadKind.INSERT_SEARCH / INSERT_DELETE_SEARCH),
writing the insert and search-query streams out as one file per step and
computing one groundtruth checkpoint per step.

Its config can be loaded from a plain `.ini` file instead of constructed
in Python (ini_config.py: load_streaming_organizer_config) -- see
workload/example/streaming_workload.example.ini.

Call `configure_logging()` (logging_utils.py) once at the start of a
script to see progress/debug output from this package.
"""

from arachne.workload.clustering import ClusterAssignment, KMeansClusterAssigner
from arachne.workload.formats import (
    OutputFormat,
    PoolWriter,
    SourceFormat,
    SourceVectorReader,
    VecsReader,
    XBinHeader,
    XBinReader,
    XBinWriter,
    open_source_reader,
    read_id_list,
    write_groundtruth,
    write_id_list,
)
from arachne.workload.groundtruth import (
    ComputeDevice,
    DistanceMetric,
    GroundTruthComputer,
    GroundTruthResult,
)
from arachne.workload.ini_config import RunSettings, load_streaming_organizer_config
from arachne.workload.logging_utils import configure_logging, get_logger
from arachne.workload.manifest import ClusterRangeInfo, StreamingWorkloadManifest
from arachne.workload.organizer import (
    StreamingOrganizerConfig,
    StreamingWorkloadOrganizer,
    step_file_path,
)
from arachne.workload.pool_split import ClusterRange, PoolRowOrder, PoolSizes, PoolSplit, PoolSplitter
from arachne.workload.streaming import (
    SegmentLocality,
    StreamingConfig,
    StreamingPlan,
    WorkloadKind,
    compute_checkpoint_groundtruth,
)

__all__ = [
    "ClusterAssignment",
    "KMeansClusterAssigner",
    "OutputFormat",
    "PoolWriter",
    "SourceFormat",
    "SourceVectorReader",
    "VecsReader",
    "XBinHeader",
    "XBinReader",
    "XBinWriter",
    "open_source_reader",
    "read_id_list",
    "write_groundtruth",
    "write_id_list",
    "ComputeDevice",
    "DistanceMetric",
    "GroundTruthComputer",
    "GroundTruthResult",
    "RunSettings",
    "load_streaming_organizer_config",
    "configure_logging",
    "get_logger",
    "ClusterRangeInfo",
    "StreamingWorkloadManifest",
    "StreamingOrganizerConfig",
    "StreamingWorkloadOrganizer",
    "step_file_path",
    "ClusterRange",
    "PoolRowOrder",
    "PoolSizes",
    "PoolSplit",
    "PoolSplitter",
    "SegmentLocality",
    "StreamingConfig",
    "StreamingPlan",
    "WorkloadKind",
    "compute_checkpoint_groundtruth",
]
