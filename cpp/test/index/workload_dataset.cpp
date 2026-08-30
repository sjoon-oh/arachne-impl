#include "workload_dataset.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace arachne::testtools {

namespace fs = std::filesystem;

namespace {

std::ifstream OpenBinary(const fs::path& path, const char* who) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error(std::string(who) + ": cannot open " + path.string());
	return f;
}

// "step_00042.u8bin" -> 42. Throws on anything not matching organizer.py's
// step_file_path() naming convention.
std::size_t ParseStepNumber(const fs::path& file) {
	std::string name = file.stem().string();
	static constexpr char kPrefix[] = "step_";
	constexpr std::size_t kPrefixLen = sizeof(kPrefix) - 1;
	if (name.size() <= kPrefixLen || name.compare(0, kPrefixLen, kPrefix) != 0) {
		throw std::runtime_error("workload_dataset: unexpected step filename: " + file.string());
	}
	return static_cast<std::size_t>(std::stoull(name.substr(kPrefixLen)));
}

// The one file directly under `dir` whose stem (filename without extension)
// equals `stem` -- used for base_pool.*/eval_query_pool.*, whose extension
// depends on dtype (see xbin_extension_for_dtype in formats.py) but whose
// stem is fixed.
fs::path FindSingleFileWithStem(const fs::path& dir, const std::string& stem, const char* who) {
	std::vector<fs::path> matches;
	for (const auto& entry : fs::directory_iterator(dir)) {
		if (entry.is_regular_file() && entry.path().stem().string() == stem) matches.push_back(entry.path());
	}
	if (matches.empty()) throw std::runtime_error(std::string(who) + ": no '" + stem + ".*' found under " + dir.string());
	if (matches.size() > 1) {
		throw std::runtime_error(std::string(who) + ": multiple '" + stem + ".*' files found under " + dir.string());
	}
	return matches.front();
}

}  // namespace

XBinHeader ReadXBinHeader(const fs::path& path) {
	std::ifstream f = OpenBinary(path, "ReadXBinHeader");
	std::uint32_t header[2] = {0, 0};
	f.read(reinterpret_cast<char*>(header), sizeof(header));
	if (!f) throw std::runtime_error("ReadXBinHeader: truncated header in " + path.string());
	return XBinHeader{static_cast<std::size_t>(header[0]), header[1]};
}

XBinBlock ReadXBinFile(const fs::path& path, std::size_t element_bytes, std::size_t max_rows) {
	std::ifstream f = OpenBinary(path, "ReadXBinFile");
	std::uint32_t header[2] = {0, 0};
	f.read(reinterpret_cast<char*>(header), sizeof(header));
	if (!f) throw std::runtime_error("ReadXBinFile: truncated header in " + path.string());

	XBinBlock block;
	block.file_count = header[0];
	block.dim = header[1];
	block.count = (max_rows == 0) ? block.file_count : std::min(max_rows, block.file_count);

	const std::size_t row_bytes = static_cast<std::size_t>(block.dim) * element_bytes;
	block.data.resize(block.count * row_bytes);
	if (!block.data.empty()) {
		f.read(reinterpret_cast<char*>(block.data.data()), static_cast<std::streamsize>(block.data.size()));
		if (!f) throw std::runtime_error("ReadXBinFile: short read from " + path.string());
	}
	return block;
}

std::vector<std::uint64_t> ReadIdList(const fs::path& path) {
	std::ifstream f = OpenBinary(path, "ReadIdList");
	std::uint32_t count = 0;
	f.read(reinterpret_cast<char*>(&count), sizeof(count));
	if (!f) throw std::runtime_error("ReadIdList: truncated header in " + path.string());

	std::vector<std::int32_t> raw(count);
	if (count > 0) {
		f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(std::int32_t)));
		if (!f) throw std::runtime_error("ReadIdList: short read from " + path.string());
	}
	return std::vector<std::uint64_t>(raw.begin(), raw.end());
}

GroundTruth ReadGroundTruth(const fs::path& path) {
	std::ifstream f = OpenBinary(path, "ReadGroundTruth");
	std::uint32_t header[2] = {0, 0};
	f.read(reinterpret_cast<char*>(header), sizeof(header));
	if (!f) throw std::runtime_error("ReadGroundTruth: truncated header in " + path.string());

	GroundTruth gt;
	gt.num_queries = header[0];
	gt.k = header[1];
	const std::size_t n = static_cast<std::size_t>(gt.num_queries) * gt.k;
	gt.ids.resize(n);
	gt.dists.resize(n);
	if (n > 0) {
		f.read(reinterpret_cast<char*>(gt.ids.data()), static_cast<std::streamsize>(n * sizeof(std::int32_t)));
		if (!f) throw std::runtime_error("ReadGroundTruth: short id read from " + path.string());
		f.read(reinterpret_cast<char*>(gt.dists.data()), static_cast<std::streamsize>(n * sizeof(float)));
		if (!f) throw std::runtime_error("ReadGroundTruth: short dist read from " + path.string());
	}
	return gt;
}

WorkloadLayout DiscoverWorkloadLayout(const fs::path& root) {
	if (!fs::is_directory(root)) throw std::runtime_error("DiscoverWorkloadLayout: not a directory: " + root.string());

	WorkloadLayout layout;
	layout.root = root;
	layout.base_pool_path = FindSingleFileWithStem(root, "base_pool", "DiscoverWorkloadLayout");
	layout.eval_query_pool_path = FindSingleFileWithStem(root, "eval_query_pool", "DiscoverWorkloadLayout");
	layout.pool_extension = layout.base_pool_path.extension().string();

	XBinHeader base_header = ReadXBinHeader(layout.base_pool_path);
	layout.dim = base_header.dim;
	layout.base_pool_count = base_header.count;

	layout.insert_dir = root / "insert";
	layout.search_dir = root / "search_query";
	layout.delete_dir = root / "delete";
	layout.groundtruth_dir = root / "groundtruth";

	std::size_t max_step = 0;
	bool any_insert_step = false;
	for (const auto& entry : fs::directory_iterator(layout.insert_dir)) {
		if (!entry.is_regular_file()) continue;
		any_insert_step = true;
		max_step = std::max(max_step, ParseStepNumber(entry.path()));
	}
	if (!any_insert_step) throw std::runtime_error("DiscoverWorkloadLayout: no step files under " + layout.insert_dir.string());
	layout.num_steps = max_step;

	if (fs::is_directory(layout.groundtruth_dir)) {
		for (const auto& entry : fs::directory_iterator(layout.groundtruth_dir)) {
			if (entry.is_regular_file()) layout.checkpoint_steps.push_back(ParseStepNumber(entry.path()));
		}
	}
	std::sort(layout.checkpoint_steps.begin(), layout.checkpoint_steps.end());

	return layout;
}

fs::path StepFilePath(const fs::path& dir, std::size_t step, const std::string& suffix) {
	char buf[32];
	std::snprintf(buf, sizeof(buf), "step_%05zu", step);
	return dir / (std::string(buf) + suffix);
}

ActiveIdTracker::ActiveIdTracker(std::size_t num_base) : num_base_(num_base) {
	// Ascending-hint insert: each new key is the largest seen so far, which
	// is the amortized-O(1)-per-insert case for std::set::insert(hint, ...).
	for (std::size_t i = 0; i < num_base; ++i) active_.insert(active_.end(), i);
}

void ActiveIdTracker::ApplyDelete(const std::vector<std::uint64_t>& global_ids) {
	for (std::uint64_t id : global_ids) active_.erase(id);
}

void ActiveIdTracker::ApplyInsertRange(std::size_t begin, std::size_t end) {
	for (std::size_t j = begin; j < end; ++j) active_.insert(active_.end(), num_base_ + j);
}

std::vector<std::uint64_t> ActiveIdTracker::SortedSnapshot() const {
	return std::vector<std::uint64_t>(active_.begin(), active_.end());
}

double MeanRecallAtK(const std::vector<std::vector<std::uint64_t>>& predicted,
		const std::vector<std::vector<std::uint64_t>>& truth) {
	double sum = 0.0;
	std::size_t counted = 0;
	for (std::size_t q = 0; q < truth.size(); ++q) {
		if (truth[q].empty()) continue;
		std::unordered_set<std::uint64_t> truth_set(truth[q].begin(), truth[q].end());
		std::size_t hit = 0;
		if (q < predicted.size()) {
			for (std::uint64_t id : predicted[q]) hit += truth_set.count(id);
		}
		sum += static_cast<double>(hit) / static_cast<double>(truth_set.size());
		++counted;
	}
	return counted == 0 ? 0.0 : sum / static_cast<double>(counted);
}

}  // namespace arachne::testtools
