#pragma once

// I/O and bookkeeping for replaying a python/stream/workload/-generated
// streaming ANN workload (see that package's organizer.py) against a C++
// index, driven by hnsw_workload_compare.cpp. Deliberately independent of
// any Arachne core/adapter type -- this file only knows about the on-disk
// layout, not about what reads it.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace arachne::testtools {

struct XBinHeader {
	std::size_t count = 0;
	std::uint32_t dim = 0;
};

// Reads just the [uint32 count][uint32 dim] header -- cheap, used to size
// capacity/plan steps without loading any vector data.
XBinHeader ReadXBinHeader(const std::filesystem::path& path);

// Up to `max_rows` rows (0 = every row the file declares) of one xbin
// file's raw element bytes, read starting right after the header.
// `element_bytes` is the caller-known per-element byte size (e.g. 1 for
// uint8/int8) -- xbin's own header has no dtype field (see
// python/stream/workload/formats.py's module doc comment), so the caller
// must already know it (matches how organizer.py's own StreamingOrganizerConfig
// takes dtype as a separate, explicit field).
struct XBinBlock {
	std::vector<std::byte> data;  // count * dim * element_bytes bytes, row-major
	std::uint32_t dim = 0;
	std::size_t count = 0;      // rows actually loaded into `data`
	std::size_t file_count = 0;  // rows the file's header declares (>= count if max_rows truncated)
};
XBinBlock ReadXBinFile(const std::filesystem::path& path, std::size_t element_bytes, std::size_t max_rows = 0);

// A delete-step ".ids" file: [uint32 count][int32 id]*count. These ids are
// *global ids* in ActiveIdTracker's 0-indexed space below (organizer.py
// draws delete_ids directly from its own `active_ids` set -- see
// organizer.py's run()) -- not row indices into any pool file.
std::vector<std::uint64_t> ReadIdList(const std::filesystem::path& path);

// A checkpoint groundtruth file: [uint32 num_queries][uint32 k], then the
// (num_queries, k) int32 neighbor-id matrix, then the (num_queries, k)
// float32 distance matrix (formats.py's write_groundtruth(), XBIN layout).
//
// IMPORTANT: ids[q*k+r] is a *position* in that checkpoint's sorted,
// deduplicated active-id set, not a global id itself -- see
// ActiveIdTracker's doc comment below for why, and how to translate it.
struct GroundTruth {
	std::uint32_t num_queries = 0;
	std::uint32_t k = 0;
	std::vector<std::int32_t> ids;  // row-major [num_queries][k]
	std::vector<float> dists;       // row-major [num_queries][k]

	std::int32_t At(std::uint32_t query, std::uint32_t rank) const { return ids[static_cast<std::size_t>(query) * k + rank]; }
};
GroundTruth ReadGroundTruth(const std::filesystem::path& path);

// Discovered layout of one organizer.py output set directory (e.g.
// .../align_workload_a_10m/set_1). `pool_extension` is whatever
// base_pool.* actually uses (e.g. ".u8bin") -- insert/ and search_query/
// step files share that same extension (organizer.py writes them with the
// same output_format.pool_file_suffix()); delete/ step files are always
// ".ids" and groundtruth/ step files always ".bin" regardless of dtype
// (OutputFormat.XBIN's own fixed suffixes). Only OutputFormat.XBIN is
// supported here (matches the sample workload) -- OutputFormat.NUMPY would
// need a real .npy/.npz parser, which this test tool does not implement.
struct WorkloadLayout {
	std::filesystem::path root;
	std::filesystem::path base_pool_path;
	std::filesystem::path eval_query_pool_path;
	std::filesystem::path insert_dir;
	std::filesystem::path search_dir;
	std::filesystem::path delete_dir;
	std::filesystem::path groundtruth_dir;
	std::string pool_extension;  // e.g. ".u8bin", including the leading dot

	std::uint32_t dim = 0;
	std::size_t base_pool_count = 0;  // base_pool.*'s own declared row count (pre-`--limit-base`)

	std::size_t num_steps = 0;                  // highest step number found under insert/
	std::vector<std::size_t> checkpoint_steps;  // ascending steps that have a groundtruth/step_*.bin file
};

// Scans `root` for the fixed organizer.py output shape and returns what it
// found. Throws std::runtime_error if base_pool.*/eval_query_pool.* aren't
// present or insert/ has no step files.
WorkloadLayout DiscoverWorkloadLayout(const std::filesystem::path& root);

// organizer.py's step_file_path(): "step_{step:05d}{suffix}", 1-indexed.
std::filesystem::path StepFilePath(const std::filesystem::path& dir, std::size_t step, const std::string& suffix);

// Mirrors python/stream/workload/organizer.py's `active_ids` bookkeeping
// exactly (see streaming.py's compute_checkpoint_groundtruth() doc
// comment): a checkpoint's groundtruth neighbor ids are *positions* in the
// sorted, deduplicated active-id set at that point in the run, not global
// ids themselves -- because deletions leave gaps, position i stops being
// equal to global id i as soon as anything at or before position i has ever
// been deleted. So a consumer must replay the identical insert/delete
// sequence (in order, with nothing skipped) and re-sort its own active-id
// set the same way to find out what global id a checkpoint's groundtruth
// position `i` actually refers to -- SortedSnapshot() below is exactly that
// re-sorted set at whatever point ApplyDelete()/ApplyInsertRange() calls
// have brought it to.
//
// Global id space (0-indexed, matching organizer.py exactly): 0..num_base-1
// is base pool row i; num_base+j is insert pool row j, cumulative across
// the whole run in step order (StreamingPlan::insert_range()'s own
// numbering -- see ApplyInsertRange()'s doc comment for why replaying steps
// in order is sufficient to reproduce this without reimplementing
// StreamingPlan's per-step split arithmetic). The VectorId a caller
// actually inserts into Arachne/hnswlib with is this global id plus 1 (0 is
// reserved -- Arachne's own "no anchor" sentinel, see controller.hpp).
class ActiveIdTracker {
 public:
	explicit ActiveIdTracker(std::size_t num_base);

	// `global_ids` are exactly what ReadIdList() returns for one delete step
	// -- already in this class's global-id space, no translation needed.
	void ApplyDelete(const std::vector<std::uint64_t>& global_ids);

	// [begin, end) is a step's insert rows' position *within the insert
	// pool* (i.e. a running "how many insert-pool rows have been applied so
	// far" counter the caller maintains itself, starting at 0) -- this
	// method adds num_base to each internally. Processing every step in
	// order with a simple running counter reproduces organizer.py's own
	// StreamingPlan::insert_range() cumulative numbering exactly, without
	// this class needing to know num_steps or replicate its per-step split
	// arithmetic at all.
	void ApplyInsertRange(std::size_t begin, std::size_t end);

	// Ascending snapshot of every currently-active global id -- position i
	// of this vector is what a checkpoint's GroundTruth::At(q, i) refers to
	// (once translated: see the class doc comment).
	std::vector<std::uint64_t> SortedSnapshot() const;

	// Whether `global_id` is currently tracked as active. A delete step's
	// ".ids" file was generated against the *full, untruncated* run (see
	// organizer.py) -- with --limit-base/--limit-steps in play, some of
	// those ids were never actually inserted into this run's own engine, so
	// callers must check this before forwarding a delete to the real engine
	// (an unconditional markDelete()/remove() on an unknown id throws/fails
	// rather than being a harmless no-op).
	bool IsActive(std::uint64_t global_id) const { return active_.count(global_id) != 0; }

	std::size_t size() const { return active_.size(); }

 private:
	std::size_t num_base_;
	std::set<std::uint64_t> active_;
};

// Mean recall@k across queries: for query q, |predicted[q] ∩ truth[q]| /
// |truth[q]|, averaged over every q with a non-empty truth[q] (a query with
// an empty truth is skipped entirely rather than counted as 0, since an
// empty truth here only ever means "no groundtruth slot decoded", not "no
// true neighbors exist"). `predicted`/`truth` both hold global ids (see
// ActiveIdTracker) -- translate hnswlib labels / Arachne VectorIds (both
// global id + 1) before calling this.
double MeanRecallAtK(const std::vector<std::vector<std::uint64_t>>& predicted,
		const std::vector<std::vector<std::uint64_t>>& truth);

}  // namespace arachne::testtools
