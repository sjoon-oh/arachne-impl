#!/usr/bin/env python3
"""End-to-end, runnable example for arachne.workload's streaming
organizer (StreamingWorkloadOrganizer): an insert(+delete)+search
workload with one groundtruth checkpoint recorded per step.

With no arguments, this script needs no external dataset: it builds a
small synthetic gaussian-blob xbin base dataset plus a separate synthetic
evaluation query file (two independent files, matching how real ANN
datasets ship base and query vectors separately), then runs both
workload kinds against them:

  1. WorkloadKind.INSERT_SEARCH -- pure growth, no deletes.
  2. WorkloadKind.INSERT_DELETE_SEARCH -- a fixed-size sliding window:
     FIFO-evict the oldest segment once the window fills up.

Pass --config to instead load every setting from an .ini file (see
`streaming_workload.example.ini` for the schema/an annotated template)
and point it at real dataset files, without editing this script at all:

    PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py --config my_streaming_workload.ini

Run with no arguments for the self-contained synthetic demo:

    PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py

See python/arachne/README.md for the step/checkpoint timeline diagram
this script walks through.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from arachne.workload import (
    ComputeDevice,
    DistanceMetric,
    OutputFormat,
    PoolSizes,
    SegmentLocality,
    SourceFormat,
    StreamingConfig,
    StreamingOrganizerConfig,
    StreamingWorkloadManifest,
    StreamingWorkloadOrganizer,
    WorkloadKind,
    configure_logging,
)
from arachne.workload.ini_config import load_streaming_organizer_config

OUTPUT_ROOT = Path(__file__).parent / "output"
DIM = 32
NUM_VECTORS = 50_000
NUM_CLUSTERS = 16
NUM_EVAL_QUERIES = 2_000

NUM_STEPS = 20  # the "iteration set" size (how many insert(+delete) steps)
NUM_INSERT = 10_000  # total insert-pool rows, divided into NUM_STEPS near-equal per-step segments
NUM_SEARCH = 5_000  # total search-pool rows, likewise -> a 2:1 insert:search rate
NUM_DELETE = 5_000  # total rows deleted over the run (WorkloadKind.INSERT_DELETE_SEARCH only)

# How many set_N generations the self-contained synthetic demo produces
# per workload kind. An .ini file's own [output] num_sets controls this
# instead when --config is given.
NUM_SETS = 1


def _write_xbin(path: Path, vectors: np.ndarray) -> None:
    num_vectors, dim = vectors.shape
    with open(path, "wb") as f:
        np.array([num_vectors, dim], dtype="<u4").tofile(f)
        vectors.tofile(f)


def build_synthetic_datasets(base_path: Path, eval_query_path: Path) -> None:
    """Writes a small xbin base dataset made of gaussian blobs (so k-means
    has real cluster structure to find), plus a separate eval query file
    drawn from the same blob centers."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(20, 230, size=(NUM_CLUSTERS, DIM))

    base_labels = rng.integers(0, NUM_CLUSTERS, size=NUM_VECTORS)
    base_vectors = np.clip(centers[base_labels] + rng.normal(0, 5, size=(NUM_VECTORS, DIM)), 0, 255).astype(np.uint8)
    _write_xbin(base_path, base_vectors)
    print(f"[example] wrote synthetic base dataset: {base_path} ({NUM_VECTORS} x {DIM}, uint8)")

    eval_labels = rng.integers(0, NUM_CLUSTERS, size=NUM_EVAL_QUERIES)
    eval_vectors = np.clip(
        centers[eval_labels] + rng.normal(0, 5, size=(NUM_EVAL_QUERIES, DIM)), 0, 255
    ).astype(np.uint8)
    _write_xbin(eval_query_path, eval_vectors)
    print(f"[example] wrote synthetic eval query dataset: {eval_query_path} ({NUM_EVAL_QUERIES} x {DIM}, uint8)")


def pick_groundtruth_device() -> ComputeDevice:
    try:
        import cupy  # noqa: F401

        print("[example] cupy is importable -> using ComputeDevice.GPU for ground truth")
        return ComputeDevice.GPU
    except ImportError:
        print("[example] cupy not available -> using ComputeDevice.CPU for ground truth")
        return ComputeDevice.CPU


def print_manifest_summary(manifest: StreamingWorkloadManifest) -> None:
    groundtruth_files = sorted(Path(manifest.groundtruth_dir).glob("step_*"))
    delete_files = sorted(Path(manifest.delete_dir).glob("step_*"))

    print(f"\n[example] === {manifest.workload_kind} / {manifest.set_name} ===")
    print(f"  segment_locality  : {manifest.segment_locality}  (insert_order={manifest.insert_order}, search_order={manifest.search_order})")
    print(f"  base_pool         : {manifest.base_pool_count} rows -> {manifest.base_pool_path}")
    print(f"  insert/           : {manifest.insert_pool_count} rows across {manifest.num_steps} step files -> {manifest.insert_dir}")
    print(f"  search_query/     : {manifest.search_pool_count} rows across {manifest.num_steps} step files -> {manifest.search_dir}")
    print(f"  eval queries      : {manifest.eval_query_pool_count} rows -> {manifest.eval_query_pool_path}")
    print(f"  num_steps         : {manifest.num_steps}")
    print(f"  num_delete        : {manifest.num_delete}  ({len(delete_files)} delete-id files written)")
    print(f"  checkpoint_steps  : {manifest.checkpoint_steps[:5]}{'...' if len(manifest.checkpoint_steps) > 5 else ''}")
    print(f"  checkpoint files  : {len(groundtruth_files)} groundtruth files written -> {manifest.groundtruth_dir}")

    # manifest.npz round-trips through disk with no JSON involved anywhere.
    reloaded = StreamingWorkloadManifest.from_npz(Path(manifest.base_pool_path).parent / "manifest.npz")
    assert reloaded == manifest
    print("  manifest.npz round-trip OK")


def run_generations(
    config: StreamingOrganizerConfig,
    source_dataset_path: Path,
    eval_query_dataset_path: Path,
    output_root: Path,
    num_sets: int,
) -> list[StreamingWorkloadManifest]:
    organizer = StreamingWorkloadOrganizer(config)
    manifests = []
    for _ in range(num_sets):
        manifest = organizer.run(
            source_dataset_path=source_dataset_path,
            eval_query_dataset_path=eval_query_dataset_path,
            output_root=output_root,
        )
        manifests.append(manifest)
    return manifests


def run_from_ini(config_path: Path) -> list[StreamingWorkloadManifest]:
    config, run_settings = load_streaming_organizer_config(config_path)
    return run_generations(
        config,
        run_settings.source_dataset_path,
        run_settings.eval_query_dataset_path,
        run_settings.output_root,
        run_settings.num_sets,
    )


def run_synthetic_demo() -> list[StreamingWorkloadManifest]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_path = OUTPUT_ROOT / "synthetic_source.u8bin"
    eval_query_path = OUTPUT_ROOT / "synthetic_eval_queries.u8bin"
    if not base_path.exists() or not eval_query_path.exists():
        build_synthetic_datasets(base_path, eval_query_path)
    else:
        print(f"[example] reusing existing synthetic datasets under {OUTPUT_ROOT}")

    device = pick_groundtruth_device()
    all_manifests: list[StreamingWorkloadManifest] = []

    for workload_kind in (WorkloadKind.INSERT_SEARCH, WorkloadKind.INSERT_DELETE_SEARCH):
        streaming_config = StreamingConfig(
            workload_kind=workload_kind,
            num_steps=NUM_STEPS,
            # num_delete must be 0 for INSERT_SEARCH -- see WorkloadKind
            num_delete=NUM_DELETE if workload_kind is WorkloadKind.INSERT_DELETE_SEARCH else 0,
            checkpoint_every=1,  # one groundtruth checkpoint per step
        )
        config = StreamingOrganizerConfig(
            source_format=SourceFormat.XBIN,
            dim=DIM,
            dtype=np.uint8,
            distance_metric=DistanceMetric.EUCLIDEAN,
            num_clusters=NUM_CLUSTERS,
            cluster_sample_size=10_000,
            pool_sizes=PoolSizes(
                num_base=20_000,
                num_insert=NUM_INSERT,
                num_query=NUM_SEARCH,
            ),
            segment_locality=SegmentLocality.ALIGN,  # NONALIGN/RANDOM also available -- see README.md
            streaming_config=streaming_config,
            groundtruth_k=10,
            groundtruth_device=device,
            output_format=OutputFormat.XBIN,
            random_seed=42,
        )
        manifests = run_generations(
            config, base_path, eval_query_path, OUTPUT_ROOT / "streaming_runs" / workload_kind.value, NUM_SETS
        )
        all_manifests.extend(manifests)

    return all_manifests


def main() -> None:
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="path to an .ini file (see streaming_workload.example.ini) -- if given, every "
             "setting comes from it instead of this script's built-in synthetic demo",
    )
    args = parser.parse_args()

    manifests = run_from_ini(args.config) if args.config is not None else run_synthetic_demo()

    for manifest in manifests:
        print_manifest_summary(manifest)

    print(f"\n[example] done. Inspect {OUTPUT_ROOT / 'streaming_runs'} for all generated files.")


if __name__ == "__main__":
    main()
