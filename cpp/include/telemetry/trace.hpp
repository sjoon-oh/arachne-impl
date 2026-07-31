#pragma once

// =============================================================================
// Opt-in fine-grained latency tracing for paper-grade benchmarking (module/
// feature breakdown graphs) -- entirely compiled out unless the
// ARACHNE_ENABLE_TRACING CMake option is on (see cpp/CMakeLists.txt), so a
// normal build/test run carries zero trace overhead: not just "disabled at
// runtime", the instrumentation code doesn't exist in the binary at all.
//
// Usage: ARACHNE_TRACE_SCOPE("RegionManager", "make") as the first statement
// of the function/block to time -- it measures from there until the
// enclosing scope exits (normal return, early return, or exception unwind
// all correctly stop the timer, since this is RAII underneath -- see
// TraceScopeGuard). Every distinct (module, feature) pair used at a given
// call site gets its own output file "<module>-<feature>.csv" (see
// TraceDir()/SanitizeForFilename() in trace.cpp), one row per TraceRecord.
//
// TraceCollector owns every TraceRecord collected under one (module,
// feature) name and writes them out the moment *it* is destroyed -- no
// separate registry, no explicit flush() call needed anywhere in traced
// code. In practice a TraceCollector is always either a function-local
// `static` inside the ARACHNE_TRACE_SCOPE() macro (one per call site,
// destroyed at process exit -- the right lifetime for a benchmark *run*) or
// a plain member of something like InstrumentedMutex (destroyed alongside
// its owner, e.g. RegionManager/OpScheduler).
//
// Thread-safety without new lock contention: record() is the hot path this
// facility exists to measure honestly, so it must not itself introduce
// serialization between threads that wouldn't otherwise contend. Each thread
// gets its own private ThreadBuffer, found through a thread_local cache kept
// inside threadBuffer():
//
//   thread_local cache (one per OS thread)         TraceCollector instance
//   +-------------------------------+               +----------------------+
//   |  id_ -> ThreadBuffer*         |   first-use    | id_                  |
//   |  id_ -> ThreadBuffer*         |   registration  | registry_mutex_      |
//   |  ...                          |  ------------>  | buffers_: vector<   |
//   +-------------------------------+                 |   unique_ptr<...>>  |
//                                                      +----------------------+
//
// registry_mutex_ is taken exactly once per (thread, collector) pair -- the
// first time that thread records anything through this collector, to append
// a new ThreadBuffer to buffers_ -- and never again afterward. Once a
// thread's entry sits in its own thread_local cache, later record() calls
// against that collector never touch the mutex or block on another thread.
//
// The cache is keyed by id_ -- a process-wide, monotonically increasing
// counter assigned at construction -- rather than by the TraceCollector's
// own `this` pointer, which looks like the obvious key but is not safe: a
// short-lived TraceCollector (e.g. InstrumentedMutex's member, owned by a
// short-lived RegionManager/OpScheduler test fixture constructed and
// destroyed many times in a loop) can easily be reconstructed at the very
// address a just-destroyed instance occupied. A `this`-keyed cache entry
// left over from the old instance would then alias the new one and hand
// back a ThreadBuffer* pointing at already-freed memory (that old
// instance's own buffers_, destroyed along with it) -- a genuine
// use-after-free, not just a logical mix-up. Because id_ never repeats for
// the life of the process, a stale cache entry from a destroyed collector is
// simply never looked up again -- it sits as a handful of harmless bytes in
// that thread's cache rather than a dangling pointer anyone dereferences.
// =============================================================================
#ifdef ARACHNE_ENABLE_TRACING

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace arachne::telemetry {

struct TraceRecord {
	std::uint64_t start_ns = 0;     // nanoseconds since the process-wide trace epoch (see TraceCollector::epoch())
	std::uint64_t duration_ns = 0;
	std::uint64_t thread_id = 0;    // std::hash<std::thread::id> of the recording thread -- unique-enough, not a small index
};

/// Per-<module, feature> record sink -- lifetime, on-destroy flush, and the
/// id_-keyed thread-local caching that keeps record() lock-free on the
/// steady-state path are all covered in the file-level overview above.
class TraceCollector {
 public:
	TraceCollector(std::string module, std::string feature);
	~TraceCollector();

	TraceCollector(const TraceCollector&) = delete;
	TraceCollector& operator=(const TraceCollector&) = delete;

	// Appends one record to the calling thread's own buffer. `start_ns` is
	// relative to epoch() so records from *different* TraceCollectors --
	// i.e. different output files -- can still be re-aligned on one shared
	// time axis during offline analysis.
	void record(std::uint64_t start_ns, std::uint64_t duration_ns);

	// Single steady_clock instant, fixed at first use for the whole process,
	// that every TraceCollector measures start_ns relative to -- must be
	// shared rather than each collector picking its own zero point.
	static std::chrono::steady_clock::time_point epoch();

 private:
	struct ThreadBuffer {
		std::vector<TraceRecord> records;
	};

	ThreadBuffer& threadBuffer();

	// Process-wide monotonically increasing id, assigned at construction --
	// what threadBuffer()'s thread_local cache actually keys on, *not*
	// `this` (see the file-level overview above for the use-after-free that
	// a `this`-keyed cache would open up on a reused address).
	static std::uint64_t nextId();
	std::uint64_t id_;

	std::string module_;
	std::string feature_;

	std::mutex registry_mutex_;  // guards buffers_ *registration* only -- see the file-level overview above
	std::vector<std::unique_ptr<ThreadBuffer>> buffers_;
};

/// RAII stopwatch: on destruction, records [construction, destruction) into
/// `collector`. The only thing ARACHNE_TRACE_SCOPE() ever creates directly;
/// not meant to be constructed by hand elsewhere.
class TraceScopeGuard {
 public:
	explicit TraceScopeGuard(TraceCollector& collector);
	~TraceScopeGuard();

	TraceScopeGuard(const TraceScopeGuard&) = delete;
	TraceScopeGuard& operator=(const TraceScopeGuard&) = delete;

 private:
	TraceCollector& collector_;
	std::chrono::steady_clock::time_point start_;
};

}  // namespace arachne::telemetry

#define ARACHNE_TRACE_CONCAT_INNER(a, b) a##b
#define ARACHNE_TRACE_CONCAT(a, b) ARACHNE_TRACE_CONCAT_INNER(a, b)

#define ARACHNE_TRACE_SCOPE(module, feature)                                                            \
	static ::arachne::telemetry::TraceCollector ARACHNE_TRACE_CONCAT(arachne_trace_collector_, __LINE__)( \
			module, feature);                                                                                \
	::arachne::telemetry::TraceScopeGuard ARACHNE_TRACE_CONCAT(arachne_trace_guard_, __LINE__)(            \
			ARACHNE_TRACE_CONCAT(arachne_trace_collector_, __LINE__))

#else  // !ARACHNE_ENABLE_TRACING

#define ARACHNE_TRACE_SCOPE(module, feature) \
	do {                                        \
	} while (0)

#endif  // ARACHNE_ENABLE_TRACING
