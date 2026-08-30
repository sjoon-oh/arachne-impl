#!/usr/bin/env bash
# Diagnostic-only sweep: runs arachne_hnsw_workload_compare --engine both
# (raw_hnswlib and arachne_controller in the SAME process, back to back --
# the only way to compare them without the isolated-vs-combined process
# timing anomaly noted during this investigation) across a matrix of
# (traverse/modify batch size, client threads, batch wait timeout)
# combinations, at a smaller base size than the main 1M investigation so a
# full sweep finishes in a reasonable time while keeping the same
# budget-to-dataset RATIO that was shown to force real eviction churn
# (see run_compare.sh's own GPU_BUDGET_BYTES comment).
#
# Not part of the regular build; this is throwaway investigation tooling.
# Existing core code is untouched -- this only drives
# arachne_hnsw_workload_compare with different CLI flags.
set -euo pipefail

WORKLOAD_DIR="/mnt/data/backup-olympus/workload/sift1b/align_workload_a_10m/set_1"
LIMIT_BASE=100000
LIMIT_STEPS=15          # 15 * 10,000 = 150,000 each of insert/delete/search
# Full residency at this base size was observed to need roughly base_size *
# (348MB / 1,000,000) bytes at the 1M scale investigated earlier; ~35MB here.
# Budget set to ~30% of that (the same ratio that produced regions_evicted
# in the thousands at 1M scale) to force real, sustained eviction pressure.
GPU_BUDGET_BYTES=10485760   # 10 MiB
M=16
EF_CONSTRUCTION=200
EF_SEARCH=100
SEED=100                # must match hnswlib's own default -- see Args::seed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../../build" && pwd)"
BIN="$BUILD_DIR/test/index/arachne_hnsw_workload_compare"

if [[ ! -x "$BIN" ]]; then
	echo "not built yet -- run: cmake --build \"$BUILD_DIR\" --target arachne_hnsw_workload_compare" >&2
	exit 1
fi

OUT_DIR="${1:-/tmp/arachne_batching_sweep}"
mkdir -p "$OUT_DIR"
SUMMARY_FILE="$OUT_DIR/summary.txt"
: > "$SUMMARY_FILE"

# (label, client_threads, traverse_batch_size, modify_batch_size,
#  batch_wait_timeout_us, group_merge_overlap_threshold, max_eviction_group_size)
CONFIGS=(
	"serial|1|1|1|0|0.5|1"
	"submit_concurrent_no_batch|32|1|1|0|0.5|1"
	"batched_no_wait|32|32|32|0|0.5|1"
	"batched_with_wait|32|32|32|500|0.5|1"
	"batched_with_wait_large|64|64|64|1000|0.5|1"
	"serial_with_group_eviction|1|1|1|0|0.5|10"
	"batched_with_wait_and_group_eviction|32|32|32|500|0.5|10"
)

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r label client_threads traverse_bs modify_bs wait_us group_threshold group_size <<< "$cfg"
	log="$OUT_DIR/${label}.log"
	echo "=== running: $label (client_threads=$client_threads traverse_bs=$traverse_bs modify_bs=$modify_bs wait_us=$wait_us group_threshold=$group_threshold group_size=$group_size) ==="
	"$BIN" \
		--workload-dir "$WORKLOAD_DIR" \
		--engine both \
		--limit-base "$LIMIT_BASE" \
		--limit-steps "$LIMIT_STEPS" \
		--gpu-budget-bytes "$GPU_BUDGET_BYTES" \
		--m "$M" --ef-construction "$EF_CONSTRUCTION" --ef-search "$EF_SEARCH" \
		--exec-threads 1 \
		--client-threads "$client_threads" \
		--traverse-batch-size "$traverse_bs" \
		--modify-batch-size "$modify_bs" \
		--batch-wait-timeout-us "$wait_us" \
		--group-merge-overlap-threshold "$group_threshold" \
		--max-eviction-group-size "$group_size" \
		--seed "$SEED" \
		--quiet-logs \
		> "$log" 2>&1

	{
		echo "=== $label (client_threads=$client_threads traverse_bs=$traverse_bs modify_bs=$modify_bs wait_us=$wait_us group_threshold=$group_threshold group_size=$group_size) ==="
		grep -A6 "=== raw_hnswlib summary ===" "$log" | head -7
		grep -A11 "=== arachne_controller summary ===" "$log" | head -12
		echo ""
	} >> "$SUMMARY_FILE"
	echo "  done -> $log"
done

echo ""
echo "=== all configs done, consolidated summary: $SUMMARY_FILE ==="
cat "$SUMMARY_FILE"
