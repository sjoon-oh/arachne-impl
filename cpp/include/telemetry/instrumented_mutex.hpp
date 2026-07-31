#pragma once

#include <mutex>

#ifdef ARACHNE_ENABLE_TRACING

#include "telemetry/trace.hpp"

namespace arachne::telemetry {

/// Opt-in lock-contention measurement for RegionManager::mutex_/
/// OpScheduler::mutex_ specifically (see cpp/CMakeLists.txt's
/// ARACHNE_ENABLE_TRACING option). When tracing is off, InstrumentedMutex
/// *is* std::mutex (a type alias, not a wrapper, see the bottom of this
/// file) -- zero overhead, not even an indirection.
///
/// Drop-in replacement for std::mutex (same lock()/unlock()/try_lock()
/// surface, so std::lock_guard<InstrumentedMutex>/std::unique_lock<
/// InstrumentedMutex> work unchanged) that additionally records how long
/// lock() spent *waiting* to acquire the underlying std::mutex -- lock
/// *contention*, as opposed to critical-section *hold* time (already
/// covered by wrapping the critical section itself in
/// ARACHNE_TRACE_SCOPE(), see telemetry/trace.hpp). An uncontended lock()
/// records a near-zero duration (just the underlying mutex's own
/// uncontended-path cost); a contended one records genuine wait time.
///
/// `module` is a string literal (e.g. "RegionManager") identifying the
/// owner -- output lands in "<module>-lockwait.csv", written when this
/// InstrumentedMutex (and therefore the collector_ it owns) is destroyed,
/// i.e. alongside its owner (RegionManager/OpScheduler)'s own destruction --
/// no separate flush() call needed, same reasoning as TraceCollector's own
/// doc comment.
///
/// RegionManager/OpScheduler both wait on a std::condition_variable against
/// this same mutex_ (coordinator_cv_/idle_cv_, cv_incoming_/cv_dispatch_).
/// std::condition_variable::wait() only accepts std::unique_lock<std::mutex>
/// specifically, not a template over the lock type, so both classes switch
/// to std::condition_variable_any (which does accept any BasicLockable)
/// *only* in the ARACHNE_ENABLE_TRACING build -- see their own
/// CoordinatorMutex/CoordinatorCondVar (or equivalent) type aliases,
/// switched by the same #ifdef, so a normal build keeps using the lighter
/// std::condition_variable + std::mutex pairing unchanged.
class InstrumentedMutex {
 public:
	explicit InstrumentedMutex(const char* module) : collector_(module, "lockwait") {}

	void lock() {
		auto start = std::chrono::steady_clock::now();
		mutex_.lock();
		auto end = std::chrono::steady_clock::now();
		std::uint64_t start_ns = static_cast<std::uint64_t>(
				std::chrono::duration_cast<std::chrono::nanoseconds>(start - TraceCollector::epoch()).count());
		std::uint64_t duration_ns =
				static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
		collector_.record(start_ns, duration_ns);
	}

	void unlock() { mutex_.unlock(); }

	bool try_lock() { return mutex_.try_lock(); }

 private:
	std::mutex mutex_;
	TraceCollector collector_;
};

}  // namespace arachne::telemetry

#else  // !ARACHNE_ENABLE_TRACING

namespace arachne::telemetry {
using InstrumentedMutex = std::mutex;
}  // namespace arachne::telemetry

#endif  // ARACHNE_ENABLE_TRACING
