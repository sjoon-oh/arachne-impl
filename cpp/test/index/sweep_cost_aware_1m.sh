#!/usr/bin/env bash
# Main-scale sweep for the replacement-policy-sweep report entry (see
# report/2026-08-27-replacement-policy-sweep.md): cost_aware only (the 5
# non-cost_aware policies were scoped out of the 1M-scale matrix after
# calibration showed unconditional-Admit policies pay a much higher fixed
# per-touch cost, unrelated to eviction pressure -- see that report entry),
# 1M base vectors, 30 steps, 30%-of-full-residency budget, sweeping
# traverse/modify batch size x (exec_threads, client_threads) pairs.
#
# Build-once optimization: RunArachneController()'s index.build() happens
# before Controller even exists, so it's completely independent of every
# knob this script sweeps -- build the 1M graph exactly once (paired with
# the one-time raw_hnswlib baseline, via --engine both, for a same-process
# build-time comparison immune to the isolated-vs-combined timing anomaly
# noted earlier in this investigation), save it, then --load-index for
# every other config instead of rebuilding.
#
# Not part of the regular build; this is throwaway investigation tooling.
# Existing core code is untouched -- this only drives
# arachne_hnsw_workload_compare with different CLI flags.
set -uo pipefail  # no -e: one config's failure/timeout shouldn't kill the sweep

WORKLOAD_DIR="/mnt/data/backup-olympus/workload/sift1b/align_workload_a_10m/set_1"
LIMIT_BASE=1000000
LIMIT_STEPS=30          # 30 * 10,000 = 300,000 each of insert/delete/search
# 30% of the full-residency footprint at 1M scale, proportionally scaled
# from the 100K-scale investigation's own established ratio (10 MiB was
# ~30% of ~35MB estimated full residency there) -- 10 MiB * 10 = 100 MiB.
GPU_BUDGET_BYTES=104857600
M=16
EF_CONSTRUCTION=200
EF_SEARCH=100
SEED=100                 # must match hnswlib's own default -- see Args::seed
GROUP_THRESHOLD=0.5
GROUP_SIZE=10             # already validated safe/beneficial, fixed across the sweep
WAIT_US=500                # nonzero so traverse/modify-batch-size sweep actually bites

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../../build" && pwd)"
BIN="$BUILD_DIR/test/index/arachne_hnsw_workload_compare"

if [[ ! -x "$BIN" ]]; then
	echo "not built yet -- run: cmake --build \"$BUILD_DIR\" --target arachne_hnsw_workload_compare" >&2
	exit 1
fi

OUT_DIR="${1:-/tmp/arachne_cost_aware_1m_sweep}"
mkdir -p "$OUT_DIR"
SUMMARY_FILE="$OUT_DIR/summary.txt"
PROGRESS_FILE="$OUT_DIR/progress.txt"
INDEX_PATH="$OUT_DIR/arachne_1m_index.bin"
: > "$PROGRESS_FILE"

# (label, traverse/modify batch size, exec_threads, client_threads)
CONFIGS=(
	"b1_t1|1|1|1"
	"b4_t1|4|1|1"
	"b8_t1|8|1|1"
	"b16_t1|16|1|1"
	"b32_t1|32|1|1"
	"b64_t1|64|1|1"
	"b128_t1|128|1|1"
	"b256_t1|256|1|1"
	"b1_t4|1|4|4"
	"b4_t4|4|4|4"
	"b8_t4|8|4|4"
	"b16_t4|16|4|4"
	"b32_t4|32|4|4"
	"b64_t4|64|4|4"
	"b128_t4|128|4|4"
	"b256_t4|256|4|4"
	"b1_t8|1|8|8"
	"b4_t8|4|8|8"
	"b8_t8|8|8|8"
	"b16_t8|16|8|8"
	"b32_t8|32|8|8"
	"b64_t8|64|8|8"
	"b128_t8|128|8|8"
	"b256_t8|256|8|8"
)

total=${#CONFIGS[@]}
i=0
for cfg in "${CONFIGS[@]}"; do
	i=$((i+1))
	IFS='|' read -r label batch exec_threads client_threads <<< "$cfg"
	log="$OUT_DIR/${label}.log"
	start_epoch=$(date +%s)
	echo "[$i/$total] running: $label (batch=$batch exec_threads=$exec_threads client_threads=$client_threads) start=$(date +%H:%M:%S)" | tee -a "$PROGRESS_FILE"

	if [[ $i -eq 1 ]]; then
		# First config: also produces the one-time raw_hnswlib baseline and
		# saves the built arachne graph for every subsequent config to load.
		engine_args=(--engine both --save-index "$INDEX_PATH")
	else
		engine_args=(--engine arachne --load-index "$INDEX_PATH")
	fi

	"$BIN" \
		--workload-dir "$WORKLOAD_DIR" \
		"${engine_args[@]}" \
		--replacement-policy cost_aware \
		--limit-base "$LIMIT_BASE" \
		--limit-steps "$LIMIT_STEPS" \
		--gpu-budget-bytes "$GPU_BUDGET_BYTES" \
		--m "$M" --ef-construction "$EF_CONSTRUCTION" --ef-search "$EF_SEARCH" \
		--exec-threads "$exec_threads" \
		--client-threads "$client_threads" \
		--traverse-batch-size "$batch" \
		--modify-batch-size "$batch" \
		--batch-wait-timeout-us "$WAIT_US" \
		--group-merge-overlap-threshold "$GROUP_THRESHOLD" \
		--max-eviction-group-size "$GROUP_SIZE" \
		--seed "$SEED" \
		--quiet-logs \
		> "$log" 2>&1
	rc=$?
	end_epoch=$(date +%s)
	elapsed=$((end_epoch - start_epoch))

	{
		echo "=== $label (batch=$batch exec_threads=$exec_threads client_threads=$client_threads) elapsed=${elapsed}s rc=$rc ==="
		if [[ $i -eq 1 ]]; then
			grep -A6 "=== raw_hnswlib summary ===" "$log" | head -7
		fi
		grep -A11 "=== arachne_controller summary ===" "$log" | head -12
		echo ""
	} >> "$SUMMARY_FILE"
	echo "[$i/$total] done: $label elapsed=${elapsed}s rc=$rc" | tee -a "$PROGRESS_FILE"
done

echo "ALL_CONFIGS_DONE" | tee -a "$PROGRESS_FILE"
echo ""
echo "=== all configs done, consolidated summary: $SUMMARY_FILE ==="
