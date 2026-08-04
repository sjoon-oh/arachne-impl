#!/usr/bin/env python3
"""Production CLI: generate one insert(+delete)+search streaming workload
from a `.ini` file (see `arachne.workload.ini_config.load_streaming_organizer_config`
for the schema, or `workload/example/streaming_workload.example.ini` for an
annotated template).

Unlike `workload/example/generate_streaming_workload.py` (a runnable
demo with a synthetic-dataset fallback), this script has no built-in
demo mode -- `--config` is required, and it's meant to be called
directly from shell scripts/CI rather than edited. `--output-root`/
`--set-name`/`--num-sets` override the ini's own `[output]` settings
without needing a second copy of the ini file, e.g. to reuse one config
across several output locations.

    PYTHONPATH=python python3 python/arachne/benchmark/dataset/generate.py \\
        --config scripts/configs/workload/my_workload.ini \\
        --output-root scripts/workload/my_dataset/my_variant \\
        --set-name set_1

Exits non-zero (and prints to stderr) on any failure, so a calling
script/CI job can detect it directly from the exit code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from arachne.workload import StreamingWorkloadOrganizer, configure_logging
from arachne.workload.ini_config import load_streaming_organizer_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="path to a streaming workload .ini file")
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="override the ini's [output] output_root",
    )
    parser.add_argument(
        "--set-name", type=str, default=None,
        help="explicit set_N name for this generation (requires exactly one generation -- "
             "i.e. num_sets resolves to 1); omit to auto-number set_1, set_2, ...",
    )
    parser.add_argument(
        "--num-sets", type=int, default=None,
        help="override the ini's [output] num_sets",
    )
    args = parser.parse_args()

    configure_logging(level=logging.INFO)

    if not args.config.exists():
        print(f"[generate] config file not found: {args.config}", file=sys.stderr)
        return 1

    config, run_settings = load_streaming_organizer_config(args.config)

    output_root = args.output_root if args.output_root is not None else run_settings.output_root
    num_sets = args.num_sets if args.num_sets is not None else run_settings.num_sets

    if args.set_name is not None and num_sets != 1:
        print(
            f"[generate] --set-name requires exactly one generation, but num_sets={num_sets} "
            f"(from {'--num-sets' if args.num_sets is not None else 'the ini file'}) -- "
            f"pass --num-sets 1 together with --set-name, or drop --set-name.",
            file=sys.stderr,
        )
        return 1

    print(f"[generate] config          = {args.config}")
    print(f"[generate] source_dataset  = {run_settings.source_dataset_path}")
    print(f"[generate] eval_query      = {run_settings.eval_query_dataset_path}")
    print(f"[generate] output_root     = {output_root}")
    print(f"[generate] num_sets        = {num_sets}")

    organizer = StreamingWorkloadOrganizer(config)
    try:
        for _ in range(num_sets):
            manifest = organizer.run(
                source_dataset_path=run_settings.source_dataset_path,
                eval_query_dataset_path=run_settings.eval_query_dataset_path,
                output_root=output_root,
                set_name=args.set_name,
            )
            print(f"[generate] wrote {manifest.set_name} -> {Path(manifest.base_pool_path).parent}")
    except Exception as e:
        print(f"[generate] workload generation failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
