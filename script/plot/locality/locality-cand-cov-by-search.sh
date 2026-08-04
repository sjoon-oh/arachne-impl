#!/bin/bash

set -euo pipefail

# Check if gnuplot is installed
if ! command -v gnuplot &> /dev/null; then
    echo "gnuplot could not be found. Please install gnuplot to run this script."
    exit 1
fi

# Resolve script directory and run from there to make input/output paths stable.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$script_dir"

lib_path="$script_dir/../lib"
output_root="$script_dir/output"
plot_file="plot-hnsw-candidate-covered-by-search.gnuplot"
size_width=6.0
size_height=2.5
color_palette="blue_gold"

mkdir -p "$output_root"

run_plot() {

    # Set default to empty string for optional arguments

    local export_name="$1"
    local output_subdir="${2:-sift1b}"
    local filename_0="${3:-}"
    local filename_1="${4:-}"
    local filename_2="${5:-}"
    local filename_3="${6:-}"
    local filename_4="${7:-}"
    local filename_5="${8:-}"
    local filename_6="${9:-}"
    local filename_7="${10:-}"
    local filename_8="${11:-}"
    local filename_9="${12:-}"
    local output_path="${output_root}/${output_subdir}"

    printf "Export base path: %s\n" "$script_dir"
    mkdir -p "$output_path"

    gnuplot -e \
            "arg_script_dir='$script_dir';
            arg_lib_dir='$lib_path';
            arg_filename_0='$filename_0';
            arg_filename_1='$filename_1';
            arg_filename_2='$filename_2';
            arg_filename_3='$filename_3';
            arg_filename_4='$filename_4';
            arg_filename_5='$filename_5';
            arg_filename_6='$filename_6';
            arg_filename_7='$filename_7';
            arg_filename_8='$filename_8';
            arg_filename_9='$filename_9';
            arg_export_base='$output_path';
            arg_export_name='$export_name';
            arg_size_width=$size_width;
            arg_size_height=$size_height;
            arg_color_palette='$color_palette';
            arg_plotfile='$plot_file'" \
            "$lib_path/template.gnuplot"
}

# SIFT1B locality candidate coverage by search plots

run_plot "sift1b-m32-efc64-efs16-cand-cov-by-search" "sift1b" \
    "hnsw-results/sift1b-m32-efc64-efs16/align_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/sift1b-m32-efc64-efs16/nonalign_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/sift1b-m32-efc64-efs16/random_workload_a_10m/set_1/search_first/arachne_trace.csv"

run_plot "sift1b-m32-efc64-efs32-cand-cov-by-search" "sift1b" \
    "hnsw-results/sift1b-m32-efc64-efs32/align_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/sift1b-m32-efc64-efs32/nonalign_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/sift1b-m32-efc64-efs32/random_workload_a_10m/set_1/search_first/arachne_trace.csv"

run_plot "sift1b-m32-efc64-efs64-cand-cov-by-search" "sift1b" \
    "hnsw-results/sift1b-m32-efc64-efs64/align_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/sift1b-m32-efc64-efs64/nonalign_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/sift1b-m32-efc64-efs64/random_workload_a_10m/set_1/search_first/arachne_trace.csv"

# SPACEV1B locality candidate coverage by search plots

run_plot "spacev1b-m32-efc64-efs16-cand-cov-by-search" "spacev1b" \
    "hnsw-results/spacev1b-m32-efc64-efs16/align_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/spacev1b-m32-efc64-efs16/nonalign_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/spacev1b-m32-efc64-efs32/random_workload_a_10m/set_1/search_first/arachne_trace.csv"

run_plot "spacev1b-m32-efc64-efs32-cand-cov-by-search" "spacev1b" \
    "hnsw-results/spacev1b-m32-efc64-efs32/align_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/spacev1b-m32-efc64-efs32/nonalign_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/spacev1b-m32-efc64-efs32/random_workload_a_10m/set_1/search_first/arachne_trace.csv"

run_plot "spacev1b-m32-efc64-efs64-cand-cov-by-search" "spacev1b" \
    "hnsw-results/spacev1b-m32-efc64-efs64/align_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/spacev1b-m32-efc64-efs64/nonalign_workload_a_10m/set_1/search_first/arachne_trace.csv" \
    "hnsw-results/spacev1b-m32-efc64-efs64/random_workload_a_10m/set_1/search_first/arachne_trace.csv"

