// A plain main()-having, non-gtest executable (same spirit as
// test/bin/full_suite_app.cpp) that replays one real, pre-generated
// streaming ANN workload -- python/stream/workload/'s organizer.py output,
// e.g. a SIFT1B "align_workload_a_10m" set -- against two engines side by
// side:
//
//   raw_hnswlib:       hnswlib::HierarchicalNSW used directly, no Arachne
//                       machinery at all (single-threaded, sequential).
//   arachne_controller: the full Arachne stack -- HnswlibIndexGpu adapter,
//                       Controller, OpScheduler, RegionManager -- exactly as
//                       a real deployment would drive it.
//
// Both replay the identical insert/delete/search_query/groundtruth sequence
// (the workload's own step files, applied in order) and report recall@k
// against the workload's own pre-computed groundtruth, plus timing, for
// each. See run_compare.sh for a parameter-editable launcher, and
// workload_dataset.hpp for the on-disk layout this reads.
//
// Concurrency: OpScheduler's worker-thread count (--exec-threads) is kept
// at 1 by default deliberately -- HnswlibIndex::modifyHost() lets hnswlib's
// own per-node locks handle real concurrent addPoint() calls (see its class
// doc comment), which is exactly what happens if >1 worker thread pulls
// Modify batches at once; that's a legitimate thing hnswlib supports, but
// it means the resulting graph is no longer a deterministic function of the
// step files alone (interleaving depends on OS thread scheduling). Pinning
// exec-threads=1 keeps the graph mutation itself serialized regardless of
// how many *client* threads submit concurrently (--client-threads), so
// recall@k stays reproducible run to run while still genuinely exercising
// OpScheduler's own concurrent-submission handling.
//
// Recall scoring caveat: --limit-base truncates the base pool below what
// the workload's own groundtruth was computed against, which makes
// recall@k numbers meaningless (the true nearest neighbors of a query
// against 10k base rows are generally not the same as against 10M) -- see
// this file's own warning printout. --limit-steps alone is always safe:
// each checkpoint's groundtruth is defined relative to the run's state at
// that step, not the run's final state, so stopping early just means fewer
// (but still individually valid) checkpoints get scored.

#include "workload_dataset.hpp"

#include <hnswlib/hnswlib.h>

#include "hnswlib_index_gpu.hpp"

#include "core/controller.hpp"
#include "core/routing_cache_hnsw.hpp"
#include "gpu/device_context.hpp"
#include "telemetry/trace.hpp"

// Diagnostic-only: raft::default_logger() is the same process-wide,
// mutex-protected sink every ARACHNE_LOG_* call in the core library writes
// through (see include/logging.hpp) -- calling its already-public
// set_level() here changes nothing about the core code itself, just lets
// this test binary silence it to isolate whether logging is contributing to
// the timing gap under investigation (see --quiet-logs below).
#include <raft/core/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace arachne;
using namespace arachne::testtools;
using arachne::index::hnsw::HnswlibIndexGpu;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Splits [0, count) into up to `num_threads` contiguous slices, each run on
// its own std::thread, joined before returning -- the concurrent-submission
// axis this binary tests (see file overview): real client threads calling
// controller.insert()/search()/remove() at the same time, independent of
// how many OpScheduler execution workers actually pull work off the queue.
template <typename Fn>
void RunConcurrently(std::size_t num_threads, std::size_t count, Fn&& fn) {
	if (count == 0) return;
	num_threads = std::max<std::size_t>(1, std::min(num_threads, count));
	if (num_threads == 1) {
		for (std::size_t i = 0; i < count; ++i) fn(i);
		return;
	}
	std::vector<std::thread> threads;
	threads.reserve(num_threads);
	std::size_t chunk = (count + num_threads - 1) / num_threads;
	for (std::size_t t = 0; t < num_threads; ++t) {
		std::size_t begin = t * chunk;
		std::size_t end = std::min(count, begin + chunk);
		if (begin >= end) continue;
		threads.emplace_back([&fn, begin, end]() {
			for (std::size_t i = begin; i < end; ++i) fn(i);
		});
	}
	for (std::thread& th : threads) th.join();
}

// Submits `count` async requests via `submit(i)` (spread across
// `client_threads`, matching RunConcurrently's slicing), then gathers every
// resulting future (also spread across `client_threads`). This is what
// actually lets OpScheduler's traverse_batch_size/modify_batch_size merge
// multiple requests into one adapter call -- unlike RunConcurrently calling
// a blocking Controller::insert()/search()/remove() per item, which caps
// how many requests can ever be simultaneously pending at `client_threads`
// regardless of batch_size (see hnsw_workload_compare.cpp's file overview
// and Controller::submitInsert()'s doc comment for why). `submit(i)` must
// return a std::future<Result>; its own value is discarded here (call sites
// needing the result read it out inside `submit` itself before returning,
// e.g. into a pre-sized output vector, the same pattern RunConcurrently's
// callers already use for predicted[q]).
template <typename Result, typename SubmitFn>
void RunAsyncBatch(std::size_t client_threads, std::size_t count, SubmitFn&& submit) {
	if (count == 0) return;
	std::vector<std::future<Result>> futures(count);
	RunConcurrently(client_threads, count, [&](std::size_t i) { futures[i] = submit(i); });
	RunConcurrently(client_threads, count, [&](std::size_t i) { futures[i].get(); });
}

// GroundTruth::At(q, r) is a *position* in `sorted_active`, not a global id
// -- see ActiveIdTracker's doc comment. -1 slots (padding for a query with
// fewer than k available truths) and out-of-range positions are skipped.
std::vector<std::vector<std::uint64_t>> TranslateGroundTruth(
		const GroundTruth& gt, const std::vector<std::uint64_t>& sorted_active) {
	std::vector<std::vector<std::uint64_t>> truth(gt.num_queries);
	for (std::uint32_t q = 0; q < gt.num_queries; ++q) {
		truth[q].reserve(gt.k);
		for (std::uint32_t r = 0; r < gt.k; ++r) {
			std::int32_t pos = gt.At(q, r);
			if (pos < 0) continue;
			std::size_t upos = static_cast<std::size_t>(pos);
			if (upos < sorted_active.size()) truth[q].push_back(sorted_active[upos]);
		}
	}
	return truth;
}

struct Args {
	std::string workload_dir = "/mnt/data/backup-olympus/workload/sift1b/align_workload_a_10m/set_1";
	std::string engine = "both";  // raw | arachne | both
	VectorDType dtype = VectorDType::UInt8;
	DistanceMetric metric = DistanceMetric::L2;
	std::size_t m = 16;
	std::size_t ef_construction = 200;
	std::size_t ef_search = 100;
	std::uint32_t stream_top_k = 10;
	std::size_t limit_steps = 0;  // 0 = every step DiscoverWorkloadLayout() found
	std::size_t limit_base = 0;   // 0 = full base pool -- see file overview's recall caveat
	std::size_t vectors_per_region = 1024;
	std::size_t gpu_data_budget_bytes = gpu::kDefaultDataPoolBytes;
	std::size_t max_execution_threads = 1;
	std::size_t client_threads = 1;
	// SchedulingConfig::traverse_batch_size/modify_batch_size passthrough --
	// default 1 matches SchedulingConfig's own default (no batching). Only
	// matters together with client_threads > 1 and the async submit* path
	// (see RunAsyncBatch): a blocking per-item call never has more than
	// client_threads requests pending at once, so batch_size beyond that
	// can't merge anything additional regardless of its value.
	std::size_t traverse_batch_size = 1;
	std::size_t modify_batch_size = 1;
	// SchedulingConfig::batch_wait_timeout, microseconds. 0 (the default)
	// means collectBatch() never waits for more arrivals: the instant the
	// queue runs dry it dispatches whatever it already has, even if that's
	// far short of batch_size -- so batch_size above is only an upper bound
	// on however many happen to already be queued at that exact instant,
	// not a guarantee. A small positive value here gives the planner a
	// brief window to let more submitInsert()/submitSearch() calls land
	// before it gives up and dispatches -- trading a little latency per
	// batch for actually reaching batch_size in practice.
	std::uint64_t batch_wait_timeout_us = 0;
	// CoordinatorConfig::group_merge_overlap_threshold/max_eviction_group_size
	// passthrough -- defaults match CoordinatorConfig's own (1 = every Anchor
	// its own singleton group, i.e. this port's original sole-ownership-only
	// reclaimability rule, unchanged unless explicitly raised).
	double group_merge_overlap_threshold = 0.5;
	std::size_t max_eviction_group_size = 1;
	// CoordinatorConfig::trigger_interval passthrough, milliseconds -- the
	// coalescing window between the first prepared candidate and batch
	// commit (see that field's own doc comment, region_manager.hpp), not an
	// intake polling interval. Default matches CoordinatorConfig's own.
	std::uint64_t trigger_interval_ms = 100;
	// Which ReplacementPolicy the arachne engine's Controller uses -- see
	// MakeReplacementPolicy() below for the accepted names. Defaults to
	// nullptr passthrough (Controller's own default, CostAwareReplacementPolicy)
	// to match every existing caller/script.
	std::string replacement_policy = "cost_aware";
	// CostAwareReplacementConfig knobs, exposed only for --replacement-policy
	// cost_aware -- each stays std::nullopt (i.e. CostAwareReplacementConfig's
	// own default) unless its flag is explicitly passed, so an unmodified
	// invocation is still byte-for-byte the same nullptr passthrough as
	// before these existed (see MakeReplacementPolicy()'s own comment).
	std::optional<std::uint64_t> cost_aware_minimum_observations;
	std::optional<std::uint64_t> cost_aware_heat_half_life_ms;
	std::optional<std::uint64_t> cost_aware_minimum_residency_ms;
	std::optional<double> cost_aware_admission_hysteresis;
	std::optional<double> cost_aware_potential_writeback_weight;
	std::optional<std::size_t> cost_aware_maximum_incremental_bytes;
	// If non-empty, RunArachneController() skips index.build() and calls
	// index.loadFrom(load_index_path) instead -- see the class's own
	// exportTo()/loadFrom() (hnswlib's saveIndex()/loadIndex() wrapped, see
	// hnswlib_index.hpp). Lets a policy/batch-size sweep across many runs
	// build the (policy- and batch-independent) base graph exactly once and
	// reuse it, instead of repeating the same deterministic, expensive work
	// every run -- see --save-index below to produce the file this reads.
	std::string load_index_path;
	// If non-empty, RunArachneController() calls index.exportTo(save_index_path)
	// right after building (real build only -- a --load-index run does not
	// re-save, since that would just be a byte-identical copy).
	std::string save_index_path;
	// Diagnostic-only, see --quiet-logs's own comment at ParseArgs.
	bool quiet_logs = false;
	bool eval_affects_policy = false;
	// hnswlib::HierarchicalNSW's own default random_seed (see its ctor in
	// thirdparty/hnswlib/hnswlib/hnswalg.h) -- matches this, not an
	// arbitrary value, because Arachne's TypedHnswEngine (hnswlib_index.cpp)
	// never passes a seed to its own HierarchicalNSW construction, so it
	// always gets this same hnswlib default. A mismatched seed here would
	// give raw_hnswlib a genuinely different random graph (different level
	// assignments -> different real build/traverse cost), confounding any
	// absolute-timing comparison between the two engines with pure RNG luck
	// rather than an architectural difference.
	unsigned seed = 100;
};

VectorDType ParseDType(const std::string& s) {
	if (s == "int8") return VectorDType::Int8;
	if (s == "uint8") return VectorDType::UInt8;
	if (s == "float16" || s == "fp16") return VectorDType::Float16;
	if (s == "float32" || s == "fp32") return VectorDType::Float32;
	std::fprintf(stderr, "unknown --dtype '%s', defaulting to uint8\n", s.c_str());
	return VectorDType::UInt8;
}

DistanceMetric ParseMetric(const std::string& s) {
	if (s == "l2" || s == "euclidean") return DistanceMetric::L2;
	if (s == "ip" || s == "inner_product") return DistanceMetric::InnerProduct;
	if (s == "cosine") return DistanceMetric::Cosine;
	std::fprintf(stderr, "unknown --metric '%s', defaulting to l2\n", s.c_str());
	return DistanceMetric::L2;
}

// nullptr means "let Controller default-construct CostAwareReplacementPolicy
// itself" -- kept distinct from explicitly naming "cost_aware" purely so
// --replacement-policy cost_aware and omitting the flag entirely are
// observably the same thing (both end up nullptr), matching every existing
// script/caller that never passed a replacement_policy argument at all. Still
// true when none of the --cost-aware-* knobs below are passed either; the
// moment even one is, this explicitly constructs a CostAwareReplacementPolicy
// instead, starting from CostAwareReplacementConfig{}'s own defaults and
// overriding only the fields that were actually given.
std::unique_ptr<ReplacementPolicy> MakeReplacementPolicy(const Args& args) {
	const std::string& name = args.replacement_policy;
	if (name == "cost_aware") {
		bool any_override = args.cost_aware_minimum_observations.has_value() ||
				args.cost_aware_heat_half_life_ms.has_value() || args.cost_aware_minimum_residency_ms.has_value() ||
				args.cost_aware_admission_hysteresis.has_value() ||
				args.cost_aware_potential_writeback_weight.has_value() ||
				args.cost_aware_maximum_incremental_bytes.has_value();
		if (!any_override) return nullptr;
		CostAwareReplacementConfig config;
		if (args.cost_aware_minimum_observations) config.minimum_observations = *args.cost_aware_minimum_observations;
		if (args.cost_aware_heat_half_life_ms) {
			config.heat_half_life = std::chrono::milliseconds(*args.cost_aware_heat_half_life_ms);
		}
		if (args.cost_aware_minimum_residency_ms) {
			config.minimum_residency = std::chrono::milliseconds(*args.cost_aware_minimum_residency_ms);
		}
		if (args.cost_aware_admission_hysteresis) config.admission_hysteresis = *args.cost_aware_admission_hysteresis;
		if (args.cost_aware_potential_writeback_weight) {
			config.potential_writeback_weight = *args.cost_aware_potential_writeback_weight;
		}
		if (args.cost_aware_maximum_incremental_bytes) {
			config.maximum_incremental_bytes = *args.cost_aware_maximum_incremental_bytes;
		}
		return std::make_unique<CostAwareReplacementPolicy>(config);
	}
	if (name == "fifo") return std::make_unique<FifoReplacementPolicy>();
	if (name == "lru") return std::make_unique<LruReplacementPolicy>();
	if (name == "lfu") return std::make_unique<LfuReplacementPolicy>();
	if (name == "clock") return std::make_unique<ClockReplacementPolicy>();
	if (name == "twoq") return std::make_unique<TwoQReplacementPolicy>();
	std::fprintf(stderr, "unknown --replacement-policy '%s', defaulting to cost_aware\n", name.c_str());
	return nullptr;
}

void PrintUsage(const char* argv0) {
	std::printf(
			"Usage: %s [options]\n"
			"  --workload-dir PATH     organizer.py output set dir (default: sample sift1b align_workload_a)\n"
			"  --engine NAME           raw | arachne | both (default both)\n"
			"  --dtype NAME            int8|uint8|float16|float32 (default uint8 -- raw_hnswlib baseline only\n"
			"                          supports uint8+l2 today, matching SIFT1B; arachne engine supports more)\n"
			"  --metric NAME           l2|ip|cosine (default l2)\n"
			"  --m N                   hnswlib M (default 16)\n"
			"  --ef-construction N     hnswlib ef_construction (default 200)\n"
			"  --ef-search N           hnswlib ef_search / setEf() (default 100 -- hnswlib's own hardcoded\n"
			"                          default is 10 if never set; both engines use --ef-search explicitly)\n"
			"  --stream-top-k N        top_k for search_query stream traffic, not scored (default 10)\n"
			"  --limit-steps N         only replay the first N steps, 0 = all (default 0; always safe --\n"
			"                          each checkpoint's groundtruth is self-consistent for a partial run)\n"
			"  --limit-base N          only load the first N base rows, 0 = full base pool (default 0;\n"
			"                          WARNING: invalidates recall@k against the provided groundtruth --\n"
			"                          only for a quick smoke test that the harness runs without crashing)\n"
			"  --vectors-per-region N  Arachne Region granularity (default 1024)\n"
			"  --gpu-budget-bytes N    Controller's gpu_data_budget_bytes (default %zu)\n"
			"  --exec-threads N        OpScheduler max_execution_threads (default 1 -- see file overview)\n"
			"  --client-threads N      concurrent submission/gather threads for insert/delete/search (default 1)\n"
			"  --traverse-batch-size N SchedulingConfig::traverse_batch_size (default 1 -- see file overview;\n"
			"                          only merges anything if client_threads > 1 too)\n"
			"  --modify-batch-size N   SchedulingConfig::modify_batch_size (default 1, same caveat)\n"
			"  --batch-wait-timeout-us N  SchedulingConfig::batch_wait_timeout in microseconds (default 0 --\n"
			"                          without this, batch_size above is only an upper bound on whatever\n"
			"                          happens to already be queued the instant the planner looks, not a\n"
			"                          guarantee; see file overview)\n"
			"  --group-merge-overlap-threshold N  CoordinatorConfig::group_merge_overlap_threshold,\n"
			"                          0.0-1.0 (default 0.5)\n"
			"  --max-eviction-group-size N  CoordinatorConfig::max_eviction_group_size (default 1 -- every\n"
			"                          Anchor its own singleton group, i.e. no behavior change from before\n"
			"                          group-based eviction existed; raise to actually enable it)\n"
			"  --trigger-interval-ms N  CoordinatorConfig::trigger_interval, milliseconds (default 100 --\n"
			"                          coalescing window between the first prepared candidate and batch\n"
			"                          commit, not an intake polling interval)\n"
			"  --replacement-policy NAME  cost_aware|fifo|lru|lfu|clock|twoq (default cost_aware, Controller's\n"
			"                          own default -- see MakeReplacementPolicy())\n"
			"  --cost-aware-min-observations N       CostAwareReplacementConfig::minimum_observations\n"
			"                          (default 1; --replacement-policy cost_aware only, ignored otherwise)\n"
			"  --cost-aware-heat-half-life-ms N      CostAwareReplacementConfig::heat_half_life, milliseconds\n"
			"                          (default 5000)\n"
			"  --cost-aware-min-residency-ms N       CostAwareReplacementConfig::minimum_residency, ms\n"
			"                          (default 0 -- no protection window)\n"
			"  --cost-aware-admission-hysteresis N   CostAwareReplacementConfig::admission_hysteresis\n"
			"                          (default 1.0 -- no margin over the best victim's density)\n"
			"  --cost-aware-writeback-weight N        CostAwareReplacementConfig::potential_writeback_weight\n"
			"                          (default 0.0 -- off)\n"
			"  --cost-aware-max-incremental-bytes N   CostAwareReplacementConfig::maximum_incremental_bytes\n"
			"                          (default 0 -- unlimited)\n"
			"  --load-index PATH       skip index.build(), call index.loadFrom(PATH) instead (arachne engine\n"
			"                          only) -- for reusing one pre-built graph across many policy/batch runs\n"
			"  --save-index PATH       after a real build (not --load-index), call index.exportTo(PATH)\n"
			"  --eval-affects-policy   let checkpoint recall queries influence promotion/eviction like real\n"
			"                          traffic (default off -- matches organizer.py's own eval queries being\n"
			"                          held out, never real stream traffic)\n"
			"  --quiet-logs            raise raft::default_logger()'s runtime level to warn (diagnostic-only,\n"
			"                          for isolating whether ARACHNE_LOG_INFO's per-call fmt::format()+\n"
			"                          mutex-protected stderr write is contributing to the timing gap under\n"
			"                          investigation -- does not touch core code, see file overview)\n"
			"  --seed N                RNG seed for raw_hnswlib's own level-assignment randomization\n"
			"                          (default 100, hnswlib's own default -- matches what Arachne's\n"
			"                          TypedHnswEngine implicitly uses; changing this makes any absolute-\n"
			"                          timing comparison between engines unfair, see file overview)\n"
			"  -h, --help              print this message\n",
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
		if (arg == "--workload-dir") {
			args.workload_dir = next("--workload-dir");
		} else if (arg == "--engine") {
			args.engine = next("--engine");
		} else if (arg == "--dtype") {
			args.dtype = ParseDType(next("--dtype"));
		} else if (arg == "--metric") {
			args.metric = ParseMetric(next("--metric"));
		} else if (arg == "--m") {
			args.m = std::stoull(next("--m"));
		} else if (arg == "--ef-construction") {
			args.ef_construction = std::stoull(next("--ef-construction"));
		} else if (arg == "--ef-search") {
			args.ef_search = std::stoull(next("--ef-search"));
		} else if (arg == "--stream-top-k") {
			args.stream_top_k = static_cast<std::uint32_t>(std::stoul(next("--stream-top-k")));
		} else if (arg == "--limit-steps") {
			args.limit_steps = std::stoull(next("--limit-steps"));
		} else if (arg == "--limit-base") {
			args.limit_base = std::stoull(next("--limit-base"));
		} else if (arg == "--vectors-per-region") {
			args.vectors_per_region = std::stoull(next("--vectors-per-region"));
		} else if (arg == "--gpu-budget-bytes") {
			args.gpu_data_budget_bytes = std::stoull(next("--gpu-budget-bytes"));
		} else if (arg == "--exec-threads") {
			args.max_execution_threads = std::stoull(next("--exec-threads"));
		} else if (arg == "--client-threads") {
			args.client_threads = std::stoull(next("--client-threads"));
		} else if (arg == "--traverse-batch-size") {
			args.traverse_batch_size = std::stoull(next("--traverse-batch-size"));
		} else if (arg == "--modify-batch-size") {
			args.modify_batch_size = std::stoull(next("--modify-batch-size"));
		} else if (arg == "--batch-wait-timeout-us") {
			args.batch_wait_timeout_us = std::stoull(next("--batch-wait-timeout-us"));
		} else if (arg == "--group-merge-overlap-threshold") {
			args.group_merge_overlap_threshold = std::stod(next("--group-merge-overlap-threshold"));
		} else if (arg == "--max-eviction-group-size") {
			args.max_eviction_group_size = std::stoull(next("--max-eviction-group-size"));
		} else if (arg == "--trigger-interval-ms") {
			args.trigger_interval_ms = std::stoull(next("--trigger-interval-ms"));
		} else if (arg == "--replacement-policy") {
			args.replacement_policy = next("--replacement-policy");
		} else if (arg == "--cost-aware-min-observations") {
			args.cost_aware_minimum_observations = std::stoull(next("--cost-aware-min-observations"));
		} else if (arg == "--cost-aware-heat-half-life-ms") {
			args.cost_aware_heat_half_life_ms = std::stoull(next("--cost-aware-heat-half-life-ms"));
		} else if (arg == "--cost-aware-min-residency-ms") {
			args.cost_aware_minimum_residency_ms = std::stoull(next("--cost-aware-min-residency-ms"));
		} else if (arg == "--cost-aware-admission-hysteresis") {
			args.cost_aware_admission_hysteresis = std::stod(next("--cost-aware-admission-hysteresis"));
		} else if (arg == "--cost-aware-writeback-weight") {
			args.cost_aware_potential_writeback_weight = std::stod(next("--cost-aware-writeback-weight"));
		} else if (arg == "--cost-aware-max-incremental-bytes") {
			args.cost_aware_maximum_incremental_bytes = std::stoull(next("--cost-aware-max-incremental-bytes"));
		} else if (arg == "--load-index") {
			args.load_index_path = next("--load-index");
		} else if (arg == "--save-index") {
			args.save_index_path = next("--save-index");
		} else if (arg == "--quiet-logs") {
			args.quiet_logs = true;
		} else if (arg == "--eval-affects-policy") {
			args.eval_affects_policy = true;
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

struct WorkloadSizing {
	std::size_t effective_num_base = 0;
	std::size_t effective_num_steps = 0;
	std::size_t total_insert = 0;
	std::size_t capacity = 0;
};

WorkloadSizing ComputeSizing(const WorkloadLayout& layout, std::size_t limit_base, std::size_t limit_steps) {
	WorkloadSizing sizing;
	sizing.effective_num_base = (limit_base == 0) ? layout.base_pool_count : std::min(limit_base, layout.base_pool_count);
	sizing.effective_num_steps = (limit_steps == 0) ? layout.num_steps : std::min(limit_steps, layout.num_steps);
	for (std::size_t step = 1; step <= sizing.effective_num_steps; ++step) {
		fs::path p = StepFilePath(layout.insert_dir, step, layout.pool_extension);
		sizing.total_insert += ReadXBinHeader(p).count;
	}
	sizing.capacity = sizing.effective_num_base + sizing.total_insert;
	return sizing;
}

struct CheckpointReport {
	std::size_t step = 0;
	double recall_at_k = 0.0;
	double eval_search_ms = 0.0;
};

struct EngineReport {
	std::string name;
	double build_ms = 0.0;
	std::size_t insert_count = 0;
	double insert_ms_total = 0.0;
	std::size_t delete_count = 0;
	double delete_ms_total = 0.0;
	std::size_t stream_search_count = 0;
	double stream_search_ms_total = 0.0;
	std::vector<CheckpointReport> checkpoints;
	bool has_controller_stats = false;
	ControllerStats controller_stats;
};

EngineReport RunRawHnswlib(const Args& args, const WorkloadLayout& layout, const WorkloadSizing& sizing) {
	if (args.dtype != VectorDType::UInt8 || args.metric != DistanceMetric::L2) {
		throw std::runtime_error(
				"RunRawHnswlib: only --dtype uint8 --metric l2 is wired up for the raw hnswlib baseline "
				"(matches SIFT1B's own dtype -- Arachne's makeEngine() supports more dtype/metric "
				"combinations, but replicating its full dispatch table for the raw baseline too was out of "
				"scope for this comparison tool)");
	}
	constexpr std::size_t kElementBytes = 1;

	EngineReport report;
	report.name = "raw_hnswlib";

	hnswlib::L2SpaceI space(layout.dim);
	hnswlib::HierarchicalNSW<int> index(&space, sizing.capacity, args.m, args.ef_construction, args.seed);
	index.setEf(args.ef_search);

	std::printf("[raw_hnswlib] building initial graph from %zu base vectors...\n", sizing.effective_num_base);
	std::fflush(stdout);
	Clock::time_point build_start = Clock::now();
	{
		ARACHNE_TRACE_SCOPE("HnswWorkloadCompareRaw", "build");
		XBinBlock base = ReadXBinFile(layout.base_pool_path, kElementBytes, sizing.effective_num_base);
		for (std::size_t i = 0; i < base.count; ++i) {
			const void* v = base.data.data() + i * static_cast<std::size_t>(base.dim) * kElementBytes;
			index.addPoint(v, static_cast<hnswlib::labeltype>(i + 1));
		}
	}
	report.build_ms = MsSince(build_start);
	std::printf("[raw_hnswlib] build done in %.1f ms\n", report.build_ms);

	ActiveIdTracker active(sizing.effective_num_base);
	std::size_t cumulative_insert = 0;
	XBinBlock eval = ReadXBinFile(layout.eval_query_pool_path, kElementBytes, 0);

	for (std::size_t step = 1; step <= sizing.effective_num_steps; ++step) {
		ARACHNE_TRACE_SCOPE("HnswWorkloadCompareRaw", "step");

		std::size_t step_delete_count = 0;
		fs::path delete_path = StepFilePath(layout.delete_dir, step, ".ids");
		if (fs::exists(delete_path)) {
			std::vector<std::uint64_t> ids = ReadIdList(delete_path);
			Clock::time_point t0 = Clock::now();
			// See ActiveIdTracker::IsActive()'s doc comment: with --limit-base/
			// --limit-steps, a delete file (generated against the full,
			// untruncated run) can reference ids this run never inserted.
			for (std::uint64_t gid : ids) {
				if (active.IsActive(gid)) index.markDelete(static_cast<hnswlib::labeltype>(gid + 1));
			}
			report.delete_ms_total += MsSince(t0);
			report.delete_count += ids.size();
			step_delete_count = ids.size();
			active.ApplyDelete(ids);
		}

		fs::path insert_path = StepFilePath(layout.insert_dir, step, layout.pool_extension);
		XBinBlock insert_block = ReadXBinFile(insert_path, kElementBytes, 0);
		{
			Clock::time_point t0 = Clock::now();
			for (std::size_t i = 0; i < insert_block.count; ++i) {
				std::uint64_t gid = sizing.effective_num_base + cumulative_insert + i;
				const void* v = insert_block.data.data() + i * static_cast<std::size_t>(insert_block.dim) * kElementBytes;
				index.addPoint(v, static_cast<hnswlib::labeltype>(gid + 1));
			}
			report.insert_ms_total += MsSince(t0);
			report.insert_count += insert_block.count;
		}
		active.ApplyInsertRange(cumulative_insert, cumulative_insert + insert_block.count);
		cumulative_insert += insert_block.count;

		fs::path search_path = StepFilePath(layout.search_dir, step, layout.pool_extension);
		if (fs::exists(search_path)) {
			XBinBlock search_block = ReadXBinFile(search_path, kElementBytes, 0);
			Clock::time_point t0 = Clock::now();
			for (std::size_t i = 0; i < search_block.count; ++i) {
				const void* v = search_block.data.data() + i * static_cast<std::size_t>(search_block.dim) * kElementBytes;
				index.searchKnnCloserFirst(v, args.stream_top_k);
			}
			report.stream_search_ms_total += MsSince(t0);
			report.stream_search_count += search_block.count;
		}

		bool is_checkpoint =
				std::binary_search(layout.checkpoint_steps.begin(), layout.checkpoint_steps.end(), step);
		if (is_checkpoint) {
			GroundTruth gt = ReadGroundTruth(StepFilePath(layout.groundtruth_dir, step, ".bin"));
			std::vector<std::uint64_t> sorted_active = active.SortedSnapshot();
			std::vector<std::vector<std::uint64_t>> truth = TranslateGroundTruth(gt, sorted_active);
			std::vector<std::vector<std::uint64_t>> predicted(gt.num_queries);

			Clock::time_point t0 = Clock::now();
			for (std::uint32_t q = 0; q < gt.num_queries; ++q) {
				const void* v = eval.data.data() + static_cast<std::size_t>(q) * eval.dim * kElementBytes;
				auto closest = index.searchKnnCloserFirst(v, gt.k);
				predicted[q].reserve(closest.size());
				for (const auto& [dist, label] : closest) predicted[q].push_back(static_cast<std::uint64_t>(label) - 1);
			}
			double eval_ms = MsSince(t0);
			double recall = MeanRecallAtK(predicted, truth);
			report.checkpoints.push_back({step, recall, eval_ms});
			std::printf(
					"[raw_hnswlib] step %zu/%zu (insert=%zu delete=%zu) -- CHECKPOINT recall@%u = %.4f "
					"(%u queries, %.1f ms)\n",
					step, sizing.effective_num_steps, insert_block.count, step_delete_count, gt.k, recall,
					gt.num_queries, eval_ms);
		} else {
			std::printf("[raw_hnswlib] step %zu/%zu done (insert=%zu delete=%zu)\n", step, sizing.effective_num_steps,
					insert_block.count, step_delete_count);
		}
		std::fflush(stdout);
	}

	return report;
}

EngineReport RunArachneController(const Args& args, const WorkloadLayout& layout, const WorkloadSizing& sizing) {
	const std::size_t element_bytes = VectorElementSize(args.dtype);

	EngineReport report;
	report.name = "arachne_controller";

	HnswlibIndexGpu index(layout.dim, args.dtype, args.metric, sizing.capacity, args.vectors_per_region, args.m,
			args.ef_construction, /*max_batch_size=*/1);
	index.setEfSearch(args.ef_search);

	Clock::time_point build_start = Clock::now();
	if (!args.load_index_path.empty()) {
		std::printf("[arachne] loading pre-built graph from '%s' (skipping build)...\n", args.load_index_path.c_str());
		std::fflush(stdout);
		ARACHNE_TRACE_SCOPE("HnswWorkloadCompareArachne", "load_index");
		index.loadFrom(args.load_index_path);
	} else {
		std::printf("[arachne] building initial graph from %zu base vectors...\n", sizing.effective_num_base);
		std::fflush(stdout);
		ARACHNE_TRACE_SCOPE("HnswWorkloadCompareArachne", "build");
		XBinBlock base = ReadXBinFile(layout.base_pool_path, element_bytes, sizing.effective_num_base);
		std::vector<VectorId> ids(base.count);
		for (std::size_t i = 0; i < base.count; ++i) ids[i] = static_cast<VectorId>(i) + 1;
		index.build(VectorBatchView{base.data.data(), base.dim, args.dtype, base.count, ids.data()});
		if (!args.save_index_path.empty()) {
			std::printf("[arachne] saving built graph to '%s'...\n", args.save_index_path.c_str());
			std::fflush(stdout);
			index.exportTo(args.save_index_path);
		}
	}
	report.build_ms = MsSince(build_start);
	std::printf("[arachne] build/load done in %.1f ms\n", report.build_ms);

	ASRoutingCacheHnsw routing_cache(layout.dim, /*initial_capacity=*/1024, /*max_tombstone_ratio=*/0.2, args.m,
			args.ef_construction, args.metric, args.dtype);
	SchedulingConfig scheduling_config;
	scheduling_config.max_execution_threads = args.max_execution_threads;
	scheduling_config.traverse_batch_size = args.traverse_batch_size;
	scheduling_config.modify_batch_size = args.modify_batch_size;
	scheduling_config.batch_wait_timeout = std::chrono::microseconds(args.batch_wait_timeout_us);
	CoordinatorConfig coordinator_config;
	coordinator_config.group_merge_overlap_threshold = args.group_merge_overlap_threshold;
	coordinator_config.max_eviction_group_size = args.max_eviction_group_size;
	coordinator_config.trigger_interval = std::chrono::milliseconds(args.trigger_interval_ms);
	Controller controller(index, routing_cache, scheduling_config, MakeReplacementPolicy(args),
			args.gpu_data_budget_bytes, gpu::kDefaultMetadataPoolBytes, gpu::kDefaultUnitBytes, nullptr,
			coordinator_config);
	index.registerAllRegions(controller);
	index.attachController(controller);

	ActiveIdTracker active(sizing.effective_num_base);
	std::size_t cumulative_insert = 0;
	XBinBlock eval = ReadXBinFile(layout.eval_query_pool_path, element_bytes, 0);

	for (std::size_t step = 1; step <= sizing.effective_num_steps; ++step) {
		ARACHNE_TRACE_SCOPE("HnswWorkloadCompareArachne", "step");

		std::size_t step_delete_count = 0;
		fs::path delete_path = StepFilePath(layout.delete_dir, step, ".ids");
		if (fs::exists(delete_path)) {
			std::vector<std::uint64_t> ids = ReadIdList(delete_path);
			// See ActiveIdTracker::IsActive()'s doc comment: with --limit-base/
			// --limit-steps, a delete file (generated against the full,
			// untruncated run) can reference ids this run never inserted --
			// filtered out before submission, not just skipped inside it, so
			// RunAsyncBatch's future count matches what's actually submitted.
			std::vector<std::uint64_t> active_ids_to_delete;
			active_ids_to_delete.reserve(ids.size());
			for (std::uint64_t gid : ids) {
				if (active.IsActive(gid)) active_ids_to_delete.push_back(gid);
			}
			Clock::time_point t0 = Clock::now();
			RunAsyncBatch<DeleteResult>(args.client_threads, active_ids_to_delete.size(), [&](std::size_t i) {
				return controller.submitRemove(static_cast<VectorId>(active_ids_to_delete[i]) + 1);
			});
			report.delete_ms_total += MsSince(t0);
			report.delete_count += ids.size();
			step_delete_count = ids.size();
			active.ApplyDelete(ids);
		}

		fs::path insert_path = StepFilePath(layout.insert_dir, step, layout.pool_extension);
		XBinBlock insert_block = ReadXBinFile(insert_path, element_bytes, 0);
		{
			Clock::time_point t0 = Clock::now();
			RunAsyncBatch<InsertResult>(args.client_threads, insert_block.count, [&](std::size_t i) {
				Record record;
				record.id = static_cast<VectorId>(sizing.effective_num_base + cumulative_insert + i) + 1;
				record.vector = VectorView{
						insert_block.data.data() + i * static_cast<std::size_t>(insert_block.dim) * element_bytes,
						insert_block.dim, args.dtype};
				// insert_block stays alive for this whole step (declared above,
				// scope spans submit and gather both) -- satisfies submitInsert()'s
				// "keep the vector alive until the future resolves" contract.
				return controller.submitInsert(record);
			});
			report.insert_ms_total += MsSince(t0);
			report.insert_count += insert_block.count;
		}
		active.ApplyInsertRange(cumulative_insert, cumulative_insert + insert_block.count);
		cumulative_insert += insert_block.count;

		// Real stream traffic -- always allowed to influence the replacement
		// policy normally (default record_for_replacement_policy=true), unlike
		// the checkpoint eval queries below.
		fs::path search_path = StepFilePath(layout.search_dir, step, layout.pool_extension);
		if (fs::exists(search_path)) {
			XBinBlock search_block = ReadXBinFile(search_path, element_bytes, 0);
			Clock::time_point t0 = Clock::now();
			RunAsyncBatch<SearchResult>(args.client_threads, search_block.count, [&](std::size_t i) {
				Query query{VectorView{search_block.data.data() + i * static_cast<std::size_t>(search_block.dim) * element_bytes,
											search_block.dim, args.dtype},
						args.stream_top_k};
				return controller.submitSearch(query);
			});
			report.stream_search_ms_total += MsSince(t0);
			report.stream_search_count += search_block.count;
		}

		bool is_checkpoint =
				std::binary_search(layout.checkpoint_steps.begin(), layout.checkpoint_steps.end(), step);
		if (is_checkpoint) {
			GroundTruth gt = ReadGroundTruth(StepFilePath(layout.groundtruth_dir, step, ".bin"));
			std::vector<std::uint64_t> sorted_active = active.SortedSnapshot();
			std::vector<std::vector<std::uint64_t>> truth = TranslateGroundTruth(gt, sorted_active);
			std::vector<std::vector<std::uint64_t>> predicted(gt.num_queries);

			// Not RunAsyncBatch here: unlike insert/delete/stream-search (result
			// discarded), each future's SearchResult must be read into
			// predicted[q] -- so submit and gather are spelled out directly
			// (the same two-phase shape RunAsyncBatch uses internally).
			Clock::time_point t0 = Clock::now();
			std::vector<std::future<SearchResult>> eval_futures(gt.num_queries);
			RunConcurrently(args.client_threads, gt.num_queries, [&](std::size_t q) {
				Query query{VectorView{eval.data.data() + q * static_cast<std::size_t>(eval.dim) * element_bytes, eval.dim,
											args.dtype},
						gt.k};
				// This is exactly the query record_for_replacement_policy exists
				// for: a measurement probe against held-out eval queries, not real
				// traffic -- see search()'s doc comment (controller.hpp) and this
				// file's --eval-affects-policy flag.
				eval_futures[q] = controller.submitSearch(query, args.eval_affects_policy);
			});
			RunConcurrently(args.client_threads, gt.num_queries, [&](std::size_t q) {
				SearchResult result = eval_futures[q].get();
				std::vector<std::uint64_t>& out = predicted[q];
				out.reserve(result.neighbors.size());
				for (const Neighbor& n : result.neighbors) out.push_back(static_cast<std::uint64_t>(n.id) - 1);
			});
			double eval_ms = MsSince(t0);
			double recall = MeanRecallAtK(predicted, truth);
			report.checkpoints.push_back({step, recall, eval_ms});
			std::printf(
					"[arachne] step %zu/%zu (insert=%zu delete=%zu) -- CHECKPOINT recall@%u = %.4f "
					"(%u queries, %.1f ms)\n",
					step, sizing.effective_num_steps, insert_block.count, step_delete_count, gt.k, recall,
					gt.num_queries, eval_ms);
		} else {
			std::printf("[arachne] step %zu/%zu done (insert=%zu delete=%zu)\n", step, sizing.effective_num_steps,
					insert_block.count, step_delete_count);
		}
		std::fflush(stdout);
	}

	controller.waitIdle();
	report.has_controller_stats = true;
	report.controller_stats = controller.stats();
	return report;
}

void PrintReport(const EngineReport& r) {
	std::printf("\n=== %s summary ===\n", r.name.c_str());
	std::printf("  build:         %.1f ms\n", r.build_ms);
	std::printf("  insert:        %zu ops, %.1f ms total (%.4f ms/op)\n", r.insert_count, r.insert_ms_total,
			r.insert_count ? r.insert_ms_total / static_cast<double>(r.insert_count) : 0.0);
	std::printf("  delete:        %zu ops, %.1f ms total (%.4f ms/op)\n", r.delete_count, r.delete_ms_total,
			r.delete_count ? r.delete_ms_total / static_cast<double>(r.delete_count) : 0.0);
	std::printf("  stream search: %zu ops, %.1f ms total (%.4f ms/op)\n", r.stream_search_count,
			r.stream_search_ms_total,
			r.stream_search_count ? r.stream_search_ms_total / static_cast<double>(r.stream_search_count) : 0.0);

	double mean_recall = 0.0;
	for (const CheckpointReport& c : r.checkpoints) mean_recall += c.recall_at_k;
	if (!r.checkpoints.empty()) mean_recall /= static_cast<double>(r.checkpoints.size());
	std::printf("  checkpoints:   %zu, mean recall@k = %.4f\n", r.checkpoints.size(), mean_recall);

	if (r.has_controller_stats) {
		const ControllerStats& s = r.controller_stats;
		std::printf(
				"  controller stats: gpu_bytes_allocated=%zu regions_promoted=%llu regions_evicted=%llu\n"
				"                     regions_written_back=%llu anchor_evictions=%llu compactions=%llu\n"
				"                     relocation_batches=%llu candidates_requeued=%llu candidates_rejected=%llu\n",
				s.gpu_bytes_allocated, static_cast<unsigned long long>(s.regions_promoted_total),
				static_cast<unsigned long long>(s.regions_evicted_total),
				static_cast<unsigned long long>(s.regions_written_back_total),
				static_cast<unsigned long long>(s.anchor_evictions_total),
				static_cast<unsigned long long>(s.compactions_total),
				static_cast<unsigned long long>(s.relocation_batches_total),
				static_cast<unsigned long long>(s.candidates_requeued_total),
				static_cast<unsigned long long>(s.candidates_rejected_total));
	}
}

void PrintComparison(const EngineReport& a, const EngineReport& b) {
	std::printf("\n=== recall@k comparison (%s vs %s) ===\n", a.name.c_str(), b.name.c_str());
	std::printf("%-8s %-16s %-16s %-10s\n", "step", a.name.c_str(), b.name.c_str(), "delta");
	std::size_t n = std::min(a.checkpoints.size(), b.checkpoints.size());
	for (std::size_t i = 0; i < n; ++i) {
		std::printf("%-8zu %-16.4f %-16.4f %-+10.4f\n", a.checkpoints[i].step, a.checkpoints[i].recall_at_k,
				b.checkpoints[i].recall_at_k, b.checkpoints[i].recall_at_k - a.checkpoints[i].recall_at_k);
	}
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

	if (args.quiet_logs) {
		raft::default_logger().set_level(rapids_logger::level_enum::warn);
		std::printf("[diag] raft::default_logger() level raised to warn (--quiet-logs)\n");
	}

	WorkloadLayout layout;
	try {
		layout = DiscoverWorkloadLayout(args.workload_dir);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "failed to discover workload layout at '%s': %s\n", args.workload_dir.c_str(), e.what());
		return 1;
	}

	WorkloadSizing sizing = ComputeSizing(layout, args.limit_base, args.limit_steps);

	std::printf(
			"=== hnsw_workload_compare ===\n"
			"  workload_dir: %s\n"
			"  engine=%s dim=%u\n"
			"  base=%zu/%zu (--limit-base=%zu) steps=%zu/%zu (--limit-steps=%zu)\n"
			"  total_insert=%zu capacity=%zu\n"
			"  M=%zu ef_construction=%zu ef_search=%zu stream_top_k=%u\n"
			"  vectors_per_region=%zu gpu_budget_bytes=%zu exec_threads=%zu client_threads=%zu "
			"eval_affects_policy=%s\n"
			"  replacement_policy=%s traverse_batch_size=%zu modify_batch_size=%zu batch_wait_timeout_us=%llu "
			"trigger_interval_ms=%llu\n",
			args.workload_dir.c_str(), args.engine.c_str(), layout.dim, sizing.effective_num_base,
			layout.base_pool_count, args.limit_base, sizing.effective_num_steps, layout.num_steps, args.limit_steps,
			sizing.total_insert, sizing.capacity, args.m, args.ef_construction, args.ef_search, args.stream_top_k,
			args.vectors_per_region, args.gpu_data_budget_bytes, args.max_execution_threads, args.client_threads,
			args.eval_affects_policy ? "true" : "false", args.replacement_policy.c_str(), args.traverse_batch_size,
			args.modify_batch_size, static_cast<unsigned long long>(args.batch_wait_timeout_us),
			static_cast<unsigned long long>(args.trigger_interval_ms));
	if (args.replacement_policy == "cost_aware") {
		// Resolved values, not just what was passed -- an unset flag still
		// prints CostAwareReplacementConfig{}'s own default here, so a sweep
		// script's logs are self-describing regardless of which knobs it
		// actually overrode.
		CostAwareReplacementConfig defaults;
		std::printf(
				"  cost_aware: min_observations=%llu heat_half_life_ms=%lld min_residency_ms=%lld "
				"admission_hysteresis=%g writeback_weight=%g max_incremental_bytes=%zu\n",
				static_cast<unsigned long long>(
						args.cost_aware_minimum_observations.value_or(defaults.minimum_observations)),
				static_cast<long long>(args.cost_aware_heat_half_life_ms.value_or(
						static_cast<std::uint64_t>(defaults.heat_half_life.count()))),
				static_cast<long long>(args.cost_aware_minimum_residency_ms.value_or(
						static_cast<std::uint64_t>(defaults.minimum_residency.count()))),
				args.cost_aware_admission_hysteresis.value_or(defaults.admission_hysteresis),
				args.cost_aware_potential_writeback_weight.value_or(defaults.potential_writeback_weight),
				args.cost_aware_maximum_incremental_bytes.value_or(defaults.maximum_incremental_bytes));
	}
	if (args.limit_base != 0 && args.limit_base < layout.base_pool_count) {
		std::printf(
				"  WARNING: --limit-base (%zu) < full base pool (%zu) -- the workload's own groundtruth was\n"
				"           computed against the FULL base pool, so recall@k below is NOT a valid measurement\n"
				"           with a truncated base (only useful as a quick crash/smoke test). Use --limit-steps\n"
				"           alone for a fast run with still-meaningful recall@k.\n",
				args.limit_base, layout.base_pool_count);
	}
	std::fflush(stdout);

	std::vector<EngineReport> reports;
	if (args.engine != "raw" && args.engine != "arachne" && args.engine != "both") {
		std::fprintf(stderr, "unknown --engine '%s' (expected raw|arachne|both)\n", args.engine.c_str());
		return 2;
	}
	if (args.engine == "raw" || args.engine == "both") reports.push_back(RunRawHnswlib(args, layout, sizing));
	if (args.engine == "arachne" || args.engine == "both") reports.push_back(RunArachneController(args, layout, sizing));

	for (const EngineReport& r : reports) PrintReport(r);
	if (reports.size() == 2) PrintComparison(reports[0], reports[1]);

	return 0;
}
