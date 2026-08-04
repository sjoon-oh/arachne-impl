"""arachne.benchmark

Orchestration scripts that run cpp/example binaries against the pools
produced by arachne.workload (base/insert/query xbin files + manifest.npz),
under the various conditions defined in experiment-plan.md /
experiment-specifics.md (e.g. E4's 2x2 residency x locality grid), and
collect their logs.

`dataset/` (implemented) is the production CLI for the workload-generation
half of this: `dataset/generate.py --config <path>` runs
StreamingWorkloadOrganizer end to end. The execution-orchestration half
(actually running cpp/example against the generated pools and collecting
logs) is not yet implemented — still a placeholder from the initial
project skeleton.
"""
