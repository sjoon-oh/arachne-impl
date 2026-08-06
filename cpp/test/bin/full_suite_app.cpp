// A plain main()-having, non-gtest counterpart to unittest/stress's
// StressIndexStage* tests -- same StressIndex adapter (test/stress, shared
// via test/CMakeLists.txt), same Controller-driven insert/search/remove
// workload, but runnable directly (`./arachne_full_suite_app`) rather than
// only through ctest, so a change can be eyeballed (timing, ControllerStats,
// pass/fail) without gtest's own output format in the way. Every parameter
// below has a sensible default; pass flags to override.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "gpu/device_context.hpp"
#include "stress_index.hpp"
#include "stress_test_support.hpp"

namespace {

using namespace arachne;
using arachne::stress::BruteForceGroundTruth;
using arachne::stress::StressIndex;
using arachne::stress::testsupport::GenerateVectors;

struct Args {
	std::uint32_t dim = 128;
	VectorDType dtype = VectorDType::Float32;
	std::size_t capacity = 20000;
	std::size_t vectors_per_region = 64;
	std::size_t num_vectors = 8000;
	std::size_t num_queries = 200;
	std::size_t num_deletes = 200;
	std::uint32_t top_k = 10;
	std::size_t gpu_data_budget_bytes = gpu::kDefaultDataPoolBytes;
	gpu::AllocationPolicy allocation_policy = gpu::AllocationPolicy::Async;
	unsigned seed = 42;
};

VectorDType ParseDType(const std::string& s) {
	if (s == "int8") return VectorDType::Int8;
	if (s == "uint8") return VectorDType::UInt8;
	if (s == "float16" || s == "fp16") return VectorDType::Float16;
	if (s == "float32" || s == "fp32") return VectorDType::Float32;
	std::fprintf(stderr, "unknown --dtype '%s', defaulting to float32\n", s.c_str());
	return VectorDType::Float32;
}

void PrintUsage(const char* argv0) {
	std::printf(
			"Usage: %s [options]\n"
			"  --dim N                  vector dimensionality (default 128)\n"
			"  --dtype NAME             int8|uint8|float16|float32 (default float32)\n"
			"  --capacity N             StressIndex's max resident vector count (default 20000)\n"
			"  --vectors-per-region N   Region granularity (default 64)\n"
			"  --num-vectors N          vectors to insert (default 8000)\n"
			"  --num-queries N          sampled search()es to verify (default 200)\n"
			"  --num-deletes N          leading ids to remove() before verifying (default 200)\n"
			"  --top-k N                neighbors per query (default 10)\n"
			"  --gpu-budget-bytes N     Controller's gpu_data_budget_bytes (default %zu)\n"
			"  --pooled                 use AllocationPolicy::Pooled instead of Normal\n"
			"  --seed N                 RNG seed (default 42)\n"
			"  -h, --help               print this message\n",
			argv0, gpu::kDefaultDataPoolBytes);
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
		if (arg == "--dim") {
			args.dim = static_cast<std::uint32_t>(std::stoul(next("--dim")));
		} else if (arg == "--dtype") {
			args.dtype = ParseDType(next("--dtype"));
		} else if (arg == "--capacity") {
			args.capacity = std::stoull(next("--capacity"));
		} else if (arg == "--vectors-per-region") {
			args.vectors_per_region = std::stoull(next("--vectors-per-region"));
		} else if (arg == "--num-vectors") {
			args.num_vectors = std::stoull(next("--num-vectors"));
		} else if (arg == "--num-queries") {
			args.num_queries = std::stoull(next("--num-queries"));
		} else if (arg == "--num-deletes") {
			args.num_deletes = std::stoull(next("--num-deletes"));
		} else if (arg == "--top-k") {
			args.top_k = static_cast<std::uint32_t>(std::stoul(next("--top-k")));
		} else if (arg == "--gpu-budget-bytes") {
			args.gpu_data_budget_bytes = std::stoull(next("--gpu-budget-bytes"));
		} else if (arg == "--pooled") {
			args.allocation_policy = gpu::AllocationPolicy::Pooled;
		} else if (arg == "--seed") {
			args.seed = static_cast<unsigned>(std::stoul(next("--seed")));
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
			return 2;
		case ParseOutcome::Ok:
			break;
	}

	std::printf(
			"=== Arachne full-suite app === dim=%u dtype=%d capacity=%zu vectors_per_region=%zu "
			"num_vectors=%zu gpu_budget_bytes=%zu policy=%s seed=%u\n",
			args.dim, static_cast<int>(args.dtype), args.capacity, args.vectors_per_region, args.num_vectors,
			args.gpu_data_budget_bytes, args.allocation_policy == gpu::AllocationPolicy::Pooled ? "Pooled" : "Normal",
			args.seed);

	StressIndex index(args.dim, args.dtype, args.capacity, args.vectors_per_region);
	ASRoutingCacheHnsw routing_cache(args.dim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2, /*M=*/16,
																	 /*ef_construction=*/200, DistanceMetric::L2, args.dtype);
	Controller controller(index, routing_cache, SchedulingConfig{}, nullptr, args.gpu_data_budget_bytes,
												 gpu::kDefaultMetadataPoolBytes, gpu::kDefaultUnitBytes, nullptr, CoordinatorConfig{},
												 args.allocation_policy);
	index.registerAllRegions(controller);

	std::mt19937 rng(args.seed);
	std::vector<std::vector<std::byte>> vectors = GenerateVectors(args.dtype, args.dim, args.num_vectors, rng);

	auto insert_start = std::chrono::steady_clock::now();
	std::size_t insert_failures = 0;
	for (std::size_t i = 0; i < args.num_vectors; ++i) {
		Record record;
		record.id = static_cast<VectorId>(i) + 1;
		record.vector = VectorView{vectors[i].data(), args.dim, args.dtype};
		if (!controller.insert(record).ok) ++insert_failures;
	}
	auto insert_end = std::chrono::steady_clock::now();
	std::printf("insert: %zu/%zu ok, %.2f ms total (%.3f ms/op)\n", args.num_vectors - insert_failures,
							args.num_vectors,
							std::chrono::duration<double, std::milli>(insert_end - insert_start).count(),
							std::chrono::duration<double, std::milli>(insert_end - insert_start).count() /
									static_cast<double>(args.num_vectors));

	std::size_t mismatches = 0;
	if (args.num_vectors > 0 && args.num_queries > 0) {
		std::uniform_int_distribution<std::size_t> pick(0, args.num_vectors - 1);
		auto search_start = std::chrono::steady_clock::now();
		for (std::size_t q = 0; q < args.num_queries; ++q) {
			std::size_t i = pick(rng);
			VectorView query_view{vectors[i].data(), args.dim, args.dtype};
			Query query{query_view, args.top_k};
			SearchResult searched = controller.search(query);
			std::vector<Neighbor> ground_truth = BruteForceGroundTruth(index, query_view, args.top_k);
			bool ok = searched.neighbors.size() == ground_truth.size();
			for (std::size_t k = 0; ok && k < ground_truth.size(); ++k) {
				ok = searched.neighbors[k].id == ground_truth[k].id;
			}
			if (!ok) ++mismatches;
		}
		auto search_end = std::chrono::steady_clock::now();
		std::printf("search: %zu/%zu queries matched brute-force ground truth, %.2f ms total (%.3f ms/op)\n",
								args.num_queries - mismatches, args.num_queries,
								std::chrono::duration<double, std::milli>(search_end - search_start).count(),
								std::chrono::duration<double, std::milli>(search_end - search_start).count() /
										static_cast<double>(args.num_queries));
	}

	std::size_t delete_failures = 0;
	std::size_t actual_deletes = std::min(args.num_deletes, args.num_vectors);
	for (std::size_t d = 0; d < actual_deletes; ++d) {
		if (!controller.remove(static_cast<VectorId>(d) + 1).ok) ++delete_failures;
	}
	std::printf("remove: %zu/%zu ok\n", actual_deletes - delete_failures, actual_deletes);

	controller.waitIdle();
	ControllerStats stats = controller.stats();
	std::printf(
			"stats: gpu_bytes_allocated=%zu regions_promoted_total=%llu regions_evicted_total=%llu "
			"regions_written_back_total=%llu anchor_evictions_total=%llu compactions_total=%llu\n",
			stats.gpu_bytes_allocated, static_cast<unsigned long long>(stats.regions_promoted_total),
			static_cast<unsigned long long>(stats.regions_evicted_total),
			static_cast<unsigned long long>(stats.regions_written_back_total),
			static_cast<unsigned long long>(stats.anchor_evictions_total),
			static_cast<unsigned long long>(stats.compactions_total));

	bool ok = insert_failures == 0 && mismatches == 0 && delete_failures == 0;
	std::printf("%s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
