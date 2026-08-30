// Build-only-and-export: constructs Arachne's HnswlibIndexGpu from a real
// organizer.py workload's base pool (python/stream/workload/, e.g. a SIFT1B
// "align_workload_a_10m" set -- see workload_dataset.hpp for the on-disk
// layout) and immediately exportTo()s the built graph to disk via
// exportTo()/loadFrom() (hnswlib's own saveIndex()/loadIndex(), wrapped --
// see hnswlib_index.hpp), then exits. No Controller, no streaming replay, no
// GPU residency involved at all -- index.build() is pure host-side HNSW
// graph construction, entirely independent of Controller/OpScheduler/
// RegionManager (confirmed while investigating hnsw_workload_compare.cpp's
// own --load-index path: build always happens before Controller even
// exists).
//
// Exists purely to pre-build large (e.g. 10M-scale) graphs once, ahead of
// time, so a later hnsw_workload_compare/sweep_cost_aware_1m.sh-style
// investigation can --load-index them instead of repeating the same
// deterministic, expensive construction work per run -- see
// cpp/test/index/report/2026-08-27-replacement-policy-sweep.md's
// "build-once-reuse" reasoning for why that matters at this data scale.
//
// Not part of the regular build; this is throwaway investigation tooling,
// same spirit as hnsw_workload_compare.cpp itself.

#include "workload_dataset.hpp"

#include "hnswlib_index_gpu.hpp"

#include "types.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace arachne;
using namespace arachne::testtools;

namespace {

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Args {
	std::string workload_dir = "/mnt/data/backup-olympus/workload/sift1b/align_workload_a_10m/set_1";
	std::string output_path;
	VectorDType dtype = VectorDType::UInt8;  // matches SIFT1B
	DistanceMetric metric = DistanceMetric::L2;
	std::size_t limit_base = 10000000;
	std::size_t m = 16;
	std::size_t ef_construction = 200;
	std::size_t vectors_per_region = 1024;
	// Element capacity HnswlibIndexGpu is constructed with -- must cover not
	// just limit_base but headroom for whatever future insert steps a later
	// --load-index consumer plans to replay against this graph (hnswlib's
	// own max_elements is fixed at construction, see hnswalg.h). 0 means
	// "compute a default": limit_base * 1.3, the same base:capacity ratio
	// this investigation's own 1M-scale build used (1,000,000 base :
	// 1,300,000 capacity, for 30 steps of 10,000 inserts each -- see
	// sweep_cost_aware_1m.sh).
	std::size_t capacity = 0;
};

void PrintUsage(const char* argv0) {
	std::printf(
			"Usage: %s [options]\n"
			"  --workload-dir PATH   organizer.py output set dir (default: sample sift1b align_workload_a)\n"
			"  --output PATH         required -- exportTo() destination for the built graph\n"
			"  --limit-base N        base vectors to build from (default 10000000 -- full SIFT1B 10M set)\n"
			"  --m N                 hnswlib M (default 16)\n"
			"  --ef-construction N   hnswlib ef_construction (default 200)\n"
			"  --vectors-per-region N  Arachne Region granularity (default 1024)\n"
			"  --capacity N          HnswlibIndexGpu element capacity (default limit_base*1.3, room for\n"
			"                        future inserts against the built graph -- see this flag's own comment)\n"
			"  -h, --help            print this message\n",
			argv0);
}

enum class ParseOutcome { Ok, HelpRequested, Error };

ParseOutcome ParseArgs(int argc, char** argv, Args& args) {
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		auto next = [&](const char* flag) -> std::string {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "%s requires a value\n", flag);
				std::exit(2);
			}
			return argv[++i];
		};
		if (arg == "--workload-dir") {
			args.workload_dir = next("--workload-dir");
		} else if (arg == "--output") {
			args.output_path = next("--output");
		} else if (arg == "--limit-base") {
			args.limit_base = std::stoull(next("--limit-base"));
		} else if (arg == "--m") {
			args.m = std::stoull(next("--m"));
		} else if (arg == "--ef-construction") {
			args.ef_construction = std::stoull(next("--ef-construction"));
		} else if (arg == "--vectors-per-region") {
			args.vectors_per_region = std::stoull(next("--vectors-per-region"));
		} else if (arg == "--capacity") {
			args.capacity = std::stoull(next("--capacity"));
		} else if (arg == "-h" || arg == "--help") {
			PrintUsage(argv[0]);
			return ParseOutcome::HelpRequested;
		} else {
			std::fprintf(stderr, "unknown argument '%s' (see --help)\n", arg.c_str());
			return ParseOutcome::Error;
		}
	}
	return ParseOutcome::Ok;
}

}  // namespace

int main(int argc, char** argv) {
	Args args;
	switch (ParseArgs(argc, argv, args)) {
		case ParseOutcome::HelpRequested:
			return 0;
		case ParseOutcome::Error:
			PrintUsage(argv[0]);
			return 2;
		case ParseOutcome::Ok:
			break;
	}
	if (args.output_path.empty()) {
		std::fprintf(stderr, "--output is required (see --help)\n");
		return 2;
	}

	WorkloadLayout layout;
	try {
		layout = DiscoverWorkloadLayout(args.workload_dir);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "failed to discover workload layout at '%s': %s\n", args.workload_dir.c_str(), e.what());
		return 1;
	}

	const std::size_t effective_num_base = std::min(args.limit_base, layout.base_pool_count);
	const std::size_t capacity =
			args.capacity != 0 ? args.capacity : static_cast<std::size_t>(effective_num_base * 1.3);

	std::printf(
			"=== build_and_export_index ===\n"
			"  workload_dir: %s\n"
			"  output: %s\n"
			"  base=%zu/%zu dim=%u M=%zu ef_construction=%zu vectors_per_region=%zu capacity=%zu\n",
			args.workload_dir.c_str(), args.output_path.c_str(), effective_num_base, layout.base_pool_count,
			layout.dim, args.m, args.ef_construction, args.vectors_per_region, capacity);
	std::fflush(stdout);

	const std::size_t element_bytes = VectorElementSize(args.dtype);
	arachne::index::hnsw::HnswlibIndexGpu index(layout.dim, args.dtype, args.metric, capacity,
			args.vectors_per_region, args.m, args.ef_construction, /*max_batch_size=*/1);

	std::printf("[build_and_export_index] building initial graph from %zu base vectors...\n", effective_num_base);
	std::fflush(stdout);
	Clock::time_point build_start = Clock::now();
	XBinBlock base = ReadXBinFile(layout.base_pool_path, element_bytes, effective_num_base);
	std::vector<VectorId> ids(base.count);
	for (std::size_t i = 0; i < base.count; ++i) ids[i] = static_cast<VectorId>(i) + 1;
	index.build(VectorBatchView{base.data.data(), base.dim, args.dtype, base.count, ids.data()});
	double build_ms = MsSince(build_start);
	std::printf("[build_and_export_index] build done in %.1f ms (%.2f min)\n", build_ms, build_ms / 60000.0);
	std::fflush(stdout);

	std::printf("[build_and_export_index] exporting to '%s'...\n", args.output_path.c_str());
	std::fflush(stdout);
	Clock::time_point export_start = Clock::now();
	index.exportTo(args.output_path);
	std::printf("[build_and_export_index] export done in %.1f ms\n", MsSince(export_start));
	std::printf("=== done ===\n");
	return 0;
}
