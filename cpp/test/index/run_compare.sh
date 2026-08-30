#!/usr/bin/env bash
# Editable launcher for arachne_hnsw_workload_compare -- change the
# variables below and rerun; no rebuild needed unless you change the C++
# itself. Run `arachne_hnsw_workload_compare --help` (or read
# hnsw_workload_compare.cpp's file overview) for what each flag means.
set -euo pipefail

# --- Parameters you'll actually want to tweak -------------------------------

WORKLOAD_DIR="/mnt/data/backup-olympus/workload/sift1b/align_workload_a_10m/set_1"
ENGINE="both"                # raw | arachne | both

# 0 = no limit. --limit-steps alone is always safe (each checkpoint's
# groundtruth is self-consistent for a partial run); --limit-base additionally
# truncates the base pool below what groundtruth was computed against, which
# makes recall@k meaningless -- only set it for a quick crash/smoke test.
LIMIT_STEPS=10
LIMIT_BASE=0

M=16
EF_CONSTRUCTION=200
EF_SEARCH=100
STREAM_TOP_K=10

VECTORS_PER_REGION=1024
GPU_BUDGET_BYTES=$((1 << 30))

# Keep at 1 -- see hnsw_workload_compare.cpp's file overview on why
# concurrent graph mutation is intentionally serialized (hnswlib's own
# per-node locks would otherwise let >1 worker thread mutate the graph at
# once, making the result depend on OS thread scheduling instead of just the
# workload's own step files).
MAX_EXECUTION_THREADS=1

# Concurrent submission/gather threads for insert/delete/search -- independent
# of MAX_EXECUTION_THREADS above. Submission itself is non-blocking (Controller
# ::submitInsert()/submitSearch()/submitRemove()), so raising this lets many
# requests be simultaneously pending, which is what TRAVERSE_BATCH_SIZE/
# MODIFY_BATCH_SIZE below actually need in order to merge anything -- with
# CLIENT_THREADS=1 there's still only ever ~1 request pending at a time
# regardless of batch size.
CLIENT_THREADS=4

# SchedulingConfig::traverse_batch_size/modify_batch_size -- how many pending
# requests OpScheduler merges into one adapter call. 1 (the default) means no
# batching regardless of CLIENT_THREADS. Raise both together with
# CLIENT_THREADS to actually see batching's effect on throughput.
TRAVERSE_BATCH_SIZE=1
MODIFY_BATCH_SIZE=1

# Without this, *_BATCH_SIZE above is only an upper bound on whatever happens
# to already be queued the instant the planner looks, not a guarantee (see
# hnsw_workload_compare.cpp's file overview) -- set a small positive value
# (microseconds) to actually reach the configured batch size in practice.
BATCH_WAIT_TIMEOUT_US=0

# 1 lets checkpoint recall queries influence promotion/eviction like real
# traffic; 0 (the workload's own intent -- eval queries are held out, never
# real stream traffic) keeps them a pure measurement probe.
EVAL_AFFECTS_POLICY=0

# hnswlib's own default random_seed -- see hnsw_workload_compare.cpp's Args::seed
# doc comment for why this must match Arachne's implicit default, not an
# arbitrary value, for a fair timing comparison between engines.
SEED=100

# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../../build" && pwd)"
BIN="$BUILD_DIR/test/index/arachne_hnsw_workload_compare"

if [[ ! -x "$BIN" ]]; then
	echo "not built yet -- run: cmake --build \"$BUILD_DIR\" --target arachne_hnsw_workload_compare" >&2
	exit 1
fi

EXTRA_ARGS=()
if [[ "$EVAL_AFFECTS_POLICY" == "1" ]]; then
	EXTRA_ARGS+=(--eval-affects-policy)
fi

exec "$BIN" \
	--workload-dir "$WORKLOAD_DIR" \
	--engine "$ENGINE" \
	--limit-steps "$LIMIT_STEPS" \
	--limit-base "$LIMIT_BASE" \
	--m "$M" \
	--ef-construction "$EF_CONSTRUCTION" \
	--ef-search "$EF_SEARCH" \
	--stream-top-k "$STREAM_TOP_K" \
	--vectors-per-region "$VECTORS_PER_REGION" \
	--gpu-budget-bytes "$GPU_BUDGET_BYTES" \
	--exec-threads "$MAX_EXECUTION_THREADS" \
	--client-threads "$CLIENT_THREADS" \
	--traverse-batch-size "$TRAVERSE_BATCH_SIZE" \
	--modify-batch-size "$MODIFY_BATCH_SIZE" \
	--batch-wait-timeout-us "$BATCH_WAIT_TIMEOUT_US" \
	--seed "$SEED" \
	"${EXTRA_ARGS[@]}"
