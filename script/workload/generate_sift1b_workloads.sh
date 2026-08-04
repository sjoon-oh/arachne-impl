#!/usr/bin/env bash
# Example: generate three SIFT1B streaming workloads that differ only in
# their insert:search rate (balanced / search-heavy / insert-heavy) -- the
# same idea as YCSB's read/write-ratio workload variants (A/B/C/...),
# applied to arachne's insert:search rate instead of a DB's read/write
# rate. Each profile's .ini lives under scripts/configs/workload/; this
# script is a reference example of driving arachne.benchmark.dataset.generate
# from a shell script, not a fixed pipeline -- copy/edit PROFILES below to
# add more ratio variants or point at a different dataset's ini files.
#
# Output lands under scripts/workload/sift1b/<profile>/set_1/ (gitignored;
# see scripts/README.md). SIFT1B's own files under
# /data2/sukjoon/datasets/SIFT1B/ are only ever read, never written.
#
# GPU selection (only matters for ini profiles with [groundtruth] device =
# gpu): set GPU_ID to pin the run to a single physical GPU, e.g.
#   GPU_ID=1 ./generate_sift1b_workloads.sh
# Leave unset to use whatever CUDA_VISIBLE_DEVICES is already set to (or
# the default device if unset).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/scripts/configs/workload"
GENERATE="$REPO_ROOT/python/arachne/benchmark/dataset/generate.py"
GPU_ID="${GPU_ID:-}"

# "profile_name:ini_file" pairs -- profile_name becomes the
# scripts/workload/sift1b/<profile_name>/ directory each one writes to.
PROFILES=(
  "align_workload_a_10m:$CONFIG_DIR/sift1b_streaming_workload_a_align.ini"
  "nonalign_workload_a_10m:$CONFIG_DIR/sift1b_streaming_workload_a_nonalign.ini"
  "random_workload_a_10m:$CONFIG_DIR/sift1b_streaming_workload_a_random.ini"
  "align_workload_b_10m:$CONFIG_DIR/sift1b_streaming_workload_b_align.ini"
  "nonalign_workload_b_10m:$CONFIG_DIR/sift1b_streaming_workload_b_nonalign.ini"
  "random_workload_b_10m:$CONFIG_DIR/sift1b_streaming_workload_b_random.ini"
  "align_workload_c_10m:$CONFIG_DIR/sift1b_streaming_workload_c_align.ini"
  "nonalign_workload_c_10m:$CONFIG_DIR/sift1b_streaming_workload_c_nonalign.ini"
  "random_workload_c_10m:$CONFIG_DIR/sift1b_streaming_workload_c_random.ini"
)

for entry in "${PROFILES[@]}"; do
  profile="${entry%%:*}"
  ini="${entry#*:}"
  out_dir="$REPO_ROOT/scripts/workload/sift1b/$profile"

  echo "=== generating $profile -> $out_dir ==="
  if [[ -n "$GPU_ID" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT/python" python3 "$GENERATE" \
      --config "$ini" \
      --output-root "$out_dir"
  else
    PYTHONPATH="$REPO_ROOT/python" python3 "$GENERATE" \
      --config "$ini" \
      --output-root "$out_dir"
  fi
done

echo "=== done: see scripts/workload/sift1b/*/set_1/ ==="
