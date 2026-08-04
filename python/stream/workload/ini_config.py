"""arachne.workload.ini_config

Loads a `StreamingOrganizerConfig` plus its run-level settings (which
files to read, where to write, how many sets to generate) from a plain
`.ini` file, using only the standard library's `configparser` -- no new
dependency, and consistent with the rest of this package's "no
third-party serialization format" stance (see formats.py/manifest.py).

This exists so a workload can be fully described in one text file and
handed to a script, instead of editing Python source every time a
different dataset/pool-size/streaming shape is needed. See
`workload/example/streaming_workload.example.ini` for a complete,
annotated example, and `workload/example/generate_streaming_workload.py
--config <path>` for how a script consumes one.

`RunSettings` holds the run-level parameters that are *not* part of
`StreamingOrganizerConfig` (mirroring how that config already excludes
`source_dataset_path`/`output_root`, which are `run()` arguments, not
workload-shape knobs): which files to read, where to write, and how many
`set_N` generations to produce.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from arachne.workload.formats import OutputFormat, SourceFormat
from arachne.workload.groundtruth import ComputeDevice, DistanceMetric
from arachne.workload.organizer import StreamingOrganizerConfig
from arachne.workload.pool_split import PoolSizes
from arachne.workload.streaming import SegmentLocality, StreamingConfig, WorkloadKind

_EnumT = TypeVar("_EnumT")


@dataclass(frozen=True)
class RunSettings:
    """Run-level parameters an ini file supplies alongside the workload
    config itself: which files to read, where to write, and how many
    `set_N` generations to produce via repeated `organizer.run()` calls.
    """

    source_dataset_path: Path
    eval_query_dataset_path: Path
    output_root: Path
    num_sets: int


def load_streaming_organizer_config(path: Path) -> tuple[StreamingOrganizerConfig, RunSettings]:
    """Loads a `StreamingOrganizerConfig` + `RunSettings` from an ini file
    (see `workload/example/streaming_workload.example.ini` for the
    schema). `[pools] num_base`/`num_insert`/`num_search` set the
    workload's ratios directly (no derivation from a per-step size --
    `[streaming] num_steps` just divides each of these, and
    `[pools] num_delete`, into that many near-equal per-step pieces, see
    streaming.StreamingPlan); `num_delete` alone may be omitted (blank ->
    0, i.e. no deletes)."""
    parser = _read_ini(path)

    streaming_config = StreamingConfig(
        workload_kind=_get_enum(parser, "streaming", "workload_kind", WorkloadKind),
        num_steps=parser.getint("streaming", "num_steps"),
        num_delete=_get_optional_int(parser, "pools", "num_delete") or 0,
        checkpoint_every=parser.getint("streaming", "checkpoint_every", fallback=1),
    )

    config = StreamingOrganizerConfig(
        source_format=_get_enum(parser, "dataset", "source_format", SourceFormat),
        dim=parser.getint("dataset", "dim"),
        dtype=np.dtype(parser.get("dataset", "dtype")),
        distance_metric=_get_enum(parser, "groundtruth", "metric", DistanceMetric),
        num_clusters=parser.getint("clustering", "num_clusters"),
        cluster_sample_size=parser.getint("clustering", "cluster_sample_size"),
        pool_sizes=PoolSizes(
            num_base=parser.getint("pools", "num_base"),
            num_insert=parser.getint("pools", "num_insert"),
            num_query=parser.getint("pools", "num_search"),
        ),
        segment_locality=_get_enum(parser, "pools", "segment_locality", SegmentLocality),
        streaming_config=streaming_config,
        groundtruth_k=parser.getint("groundtruth", "k"),
        groundtruth_device=_get_enum(parser, "groundtruth", "device", ComputeDevice),
        output_format=_get_enum(parser, "output", "format", OutputFormat),
        random_seed=parser.getint("output", "random_seed"),
    )
    return config, _load_run_settings(parser)


def _load_run_settings(parser: configparser.ConfigParser) -> RunSettings:
    return RunSettings(
        source_dataset_path=Path(parser.get("dataset", "source_dataset_path")),
        eval_query_dataset_path=Path(parser.get("dataset", "eval_query_dataset_path")),
        output_root=Path(parser.get("output", "output_root")),
        num_sets=parser.getint("output", "num_sets", fallback=1),
    )


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    read_files = parser.read(path)
    if not read_files:
        raise FileNotFoundError(f"ini config file not found or unreadable: {path}")
    return parser


def _get_enum(parser: configparser.ConfigParser, section: str, key: str, enum_cls: type[_EnumT]) -> _EnumT:
    raw = parser.get(section, key)
    try:
        return enum_cls(raw)  # type: ignore[call-arg]
    except ValueError as e:
        valid = ", ".join(member.value for member in enum_cls)  # type: ignore[attr-defined]
        raise ValueError(
            f"[{section}] {key} = {raw!r} is not valid for {enum_cls.__name__}; "
            f"expected one of: {valid}"
        ) from e


def _get_optional_int(parser: configparser.ConfigParser, section: str, key: str) -> int | None:
    """Returns None if the key is absent or left blank (e.g.
    `delete_window_steps =` with nothing after the `=`), so an ini file
    can explicitly request a config's own default instead of the caller
    having to know and repeat that default's numeric value."""
    if not parser.has_option(section, key):
        return None
    raw = parser.get(section, key).strip()
    return int(raw) if raw else None
