#pragma once

// Opt-in fine-grained latency tracing for paper-grade benchmarking (module/
// feature breakdown graphs) -- entirely compiled out unless the
// ARACHNE_ENABLE_TRACING CMake option is on (see cpp/CMakeLists.txt), so a
// normal build/test run carries zero trace overhead: not just "disabled at
// runtime", the instrumentation code doesn't exist in the binary at all.
//
// Usage: ARACHNE_TRACE_SCOPE("RegionManager", "make") as the first statement
// of the function/block to time -- it measures from there until the
// enclosing scope exits (normal return, early return, or exception unwind
// all correctly stop the timer, since this is RAII underneath). Every
// distinct (module, feature) pair used at a given call site gets its own
// output file, named "<module>-<feature>.csv" -- see TraceCollector's own
// doc comment for why this needs no explicit flush() call anywhere.
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

/// Owns every TraceRecord collected under one <module, feature> name and
/// writes them to "<module>-<feature>.csv" the moment *this instance* is
/// destroyed -- no separate registry, no explicit flush() call needed
/// anywhere in the traced code. In practice this instance is always either
/// a function-local `static` inside the ARACHNE_TRACE_SCOPE() macro (one
/// per call site, destroyed at process exit -- the right lifetime for a
/// benchmark *run*, which is the intended use of this whole facility) or
/// owned as a plain member of something like InstrumentedMutex (destroyed
/// alongside its owner, e.g. RegionManager/OpScheduler).
///
/// Thread-safety without new lock contention: recording (record()) is the
/// hot path this whole facility exists to measure honestly, so it must not
/// itself introduce serialization between threads that wouldn't otherwise
/// contend. Each thread gets its own private ThreadBuffer (looked up via a
/// thread_local cache keyed by `id_` -- see its own doc comment for why a
/// plain `this` pointer is not safe to key it by, despite looking like the
/// obvious choice) -- registry_mutex_ is only ever taken once per (thread,
/// collector) pair, the first time that thread records anything here, and
/// never again afterward. Concurrent record() calls from different threads
/// that have already registered never block each other.
class TraceCollector {
 public:
	TraceCollector(std::string module, std::string feature);
	~TraceCollector();

	TraceCollector(const TraceCollector&) = delete;
	TraceCollector& operator=(const TraceCollector&) = delete;

	// Appends one record to the calling thread's own buffer. `start_ns` is
	// relative to epoch() (see below) so records from *different*
	// TraceCollectors -- i.e. different output files -- can still be
	// re-aligned on one shared time axis during offline analysis.
	void record(std::uint64_t start_ns, std::uint64_t duration_ns);

	// A single steady_clock instant, fixed at first use for the whole
	// process, that every TraceCollector measures start_ns relative to --
	// see record()'s own doc comment for why this must be shared rather
	// than each collector picking its own zero point.
	static std::chrono::steady_clock::time_point epoch();

 private:
	struct ThreadBuffer {
		std::vector<TraceRecord> records;
	};

	ThreadBuffer& threadBuffer();

	// A process-wide monotonically increasing id, assigned once at
	// construction -- what threadBuffer()'s thread_local cache actually keys
	// on, *not* `this`. A short-lived TraceCollector (e.g. InstrumentedMutex's
	// member, owned by a short-lived RegionManager/OpScheduler test fixture
	// constructed and destroyed many times in a loop) can easily be
	// constructed again at the very address a just-destroyed one occupied --
	// a `this`-keyed cache entry from the old instance would then alias the
	// new one and hand back a ThreadBuffer* pointing at already-freed memory
	// (that old instance's own buffers_, destroyed along with it) -- a
	// genuine use-after-free, not just a logical mix-up. An id that never
	// repeats for the lifetime of the process closes this off entirely: a
	// stale cache entry from a destroyed collector is simply never looked up
	// again (its id_ died with it), left as a handful of harmless bytes in
	// that thread's cache rather than a dangling pointer anyone dereferences.
	static std::uint64_t nextId();
	std::uint64_t id_;

	std::string module_;
	std::string feature_;

	std::mutex registry_mutex_;  // guards buffers_ *registration* only -- see the class doc comment
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
