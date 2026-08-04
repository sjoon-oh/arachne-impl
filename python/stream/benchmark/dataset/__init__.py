"""arachne.benchmark.dataset

Production entry point for workload generation: `generate.py` runs
`StreamingWorkloadOrganizer` end to end from a single `.ini` file, meant
to be invoked directly by scripts/CI (see the repository's `scripts/`
directory) -- unlike `arachne.workload.example`, which is illustrative
sample code, not a stable CLI contract.
"""
