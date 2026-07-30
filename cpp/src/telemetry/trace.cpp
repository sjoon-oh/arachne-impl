#include "telemetry/trace.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unordered_map>

namespace arachne::telemetry {

namespace {

// Output directory for every "<module>-<feature>.csv" file this process
// produces -- overridable via ARACHNE_TRACE_DIR so a benchmark script can
// redirect an entire run's output without recompiling. Defaults to the
// current working directory.
std::string TraceDir() {
	const char* env = std::getenv("ARACHNE_TRACE_DIR");
	return env != nullptr ? std::string(env) : std::string(".");
}

// Sanitizes a module/feature string for safe use as a filename component --
// callers are expected to pass simple identifiers (see ARACHNE_TRACE_SCOPE's
// own doc comment), this only guards against a stray '/' turning one path
// segment into two.
std::string SanitizeForFilename(const std::string& raw) {
	std::string sanitized = raw;
	for (char& c : sanitized) {
		if (c == '/' || c == '\\') c = '_';
	}
	return sanitized;
}

}  // namespace

std::chrono::steady_clock::time_point TraceCollector::epoch() {
	static const std::chrono::steady_clock::time_point kEpoch = std::chrono::steady_clock::now();
	return kEpoch;
}

std::uint64_t TraceCollector::nextId() {
	static std::atomic<std::uint64_t> counter{0};
	return counter.fetch_add(1, std::memory_order_relaxed);
}

TraceCollector::TraceCollector(std::string module, std::string feature)
		: id_(nextId()), module_(std::move(module)), feature_(std::move(feature)) {}

TraceCollector::~TraceCollector() {
	std::string path = TraceDir() + "/" + SanitizeForFilename(module_) + "-" + SanitizeForFilename(feature_) + ".csv";
	FILE* file = std::fopen(path.c_str(), "w");
	if (file == nullptr) return;  // best-effort -- a traced program must never crash because of tracing itself

	std::fputs("start_ns,duration_ns,thread_id\n", file);
	// No lock needed here in practice (every recording thread is expected to
	// have stopped by the time this destructor runs -- see the class doc
	// comment on why this is always either a process-exit-time static or
	// tied to an owner, like RegionManager, whose own destruction already
	// implies its worker/Coordinator threads have joined), but taking
	// registry_mutex_ anyway costs nothing on this cold, one-time path and
	// removes any doubt.
	std::lock_guard<std::mutex> lock(registry_mutex_);
	for (const std::unique_ptr<ThreadBuffer>& buffer : buffers_) {
		for (const TraceRecord& record : buffer->records) {
			std::fprintf(file, "%llu,%llu,%llu\n", static_cast<unsigned long long>(record.start_ns),
									 static_cast<unsigned long long>(record.duration_ns),
									 static_cast<unsigned long long>(record.thread_id));
		}
	}
	std::fclose(file);
}

TraceCollector::ThreadBuffer& TraceCollector::threadBuffer() {
	// Keyed by id_, not `this` -- see id_'s own doc comment (trace.hpp) for
	// the use-after-free a `this`-keyed cache would open up once a
	// short-lived TraceCollector's address gets reused. One thread commonly
	// records into several different TraceCollectors (one per
	// ARACHNE_TRACE_SCOPE() call site it passes through), hence a map here
	// rather than one single thread_local buffer.
	thread_local std::unordered_map<std::uint64_t, ThreadBuffer*> cache;

	auto it = cache.find(id_);
	if (it != cache.end()) return *it->second;

	auto owned = std::make_unique<ThreadBuffer>();
	ThreadBuffer* raw = owned.get();
	{
		std::lock_guard<std::mutex> lock(registry_mutex_);
		buffers_.push_back(std::move(owned));
	}
	cache.emplace(id_, raw);
	return *raw;
}

void TraceCollector::record(std::uint64_t start_ns, std::uint64_t duration_ns) {
	ThreadBuffer& buffer = threadBuffer();
	std::uint64_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
	buffer.records.push_back(TraceRecord{start_ns, duration_ns, thread_id});
}

TraceScopeGuard::TraceScopeGuard(TraceCollector& collector)
		: collector_(collector), start_(std::chrono::steady_clock::now()) {}

TraceScopeGuard::~TraceScopeGuard() {
	auto end = std::chrono::steady_clock::now();
	std::uint64_t start_ns = static_cast<std::uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(start_ - TraceCollector::epoch()).count());
	std::uint64_t duration_ns =
			static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
	collector_.record(start_ns, duration_ns);
}

}  // namespace arachne::telemetry
