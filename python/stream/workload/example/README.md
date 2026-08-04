# `arachne.workload` example

- `generate_streaming_workload.py` — self-contained, runnable walkthrough of
  the `StreamingWorkloadOrganizer` pipeline. It fabricates its own small
  synthetic base dataset *and* a separate synthetic evaluation query file (no
  real dataset needed), then runs both `WorkloadKind.INSERT_SEARCH` and
  `WorkloadKind.INSERT_DELETE_SEARCH`, each producing one groundtruth
  checkpoint per step.
- `inspect_source_formats.py` — hand-writes an fvecs file and a SPACEV-style
  file and confirms `open_source_reader` reads them back exactly right.

```bash
PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py
```

Output is written to `example/output/`, which is gitignored — safe to delete
and re-run at any time.

The script also accepts `--config <path>` to load every setting from a
plain `.ini` file instead of the built-in synthetic demo:

- `quickstart_streaming_workload.ini` — simple, immediately runnable: it
  points at the same synthetic files the no-argument demo above already
  writes, so `--config` can be tried with no real dataset either.
- `streaming_workload.example.ini` — a fully annotated template (every
  `.ini` section explained) meant to be copied and pointed at a real dataset
  (e.g. `base.1B.u8bin` + `query.public.10K.u8bin`).

```bash
PYTHONPATH=python python3 python/arachne/workload/example/generate_streaming_workload.py \
    --config python/arachne/workload/example/quickstart_streaming_workload.ini
```

See `../../README.md` for the full pipeline explanation (ASCII diagrams,
dataset layout, the stream-query vs. evaluation-query split, "workload
generation sets", GPU-accelerated ground truth, streaming workloads, and
`.ini` config loading).
