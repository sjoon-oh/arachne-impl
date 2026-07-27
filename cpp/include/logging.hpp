#pragma once

// Arachne's logging facade: one set of macros (ARACHNE_LOG_*) used
// throughout include/src. What backs them depends on whether RAFT is linked
// (see the ARACHNE_USE_RAFT option in CMakeLists.txt):
//
//  - RAFT linked: forwards to RAFT's own logger (raft::default_logger(), a
//    rapids-logger/spdlog wrapper) so Arachne's control-plane logs and
//    RAFT's GPU-primitive logs share one sink, pattern, and level.
//  - RAFT not linked: falls back to a standalone spdlog logger.
//
// spdlog is the engine underneath either way, but the two aren't
// format-compatible: the spdlog fallback below takes fmt-style "{}"
// placeholders directly, while rapids_logger::logger::log() (what
// RAFT_LOG_* forwards to) formats with plain printf-style specifiers
// (std::snprintf internally) and silently drops "{}" as inert literal text
// instead of substituting it -- every ARACHNE_LOG_* call site in this
// codebase is written fmt-style, so the RAFT branch formats the message
// eagerly with fmt itself and hands raft's logger the already-formatted
// string via "%s", instead of requiring two different call styles at every
// call site depending on which branch is active.

#if defined(ARACHNE_WITH_RAFT)

#include <fmt/format.h>
#include <raft/core/logger.hpp>

#define ARACHNE_LOG_TRACE(...) RAFT_LOG_TRACE("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_DEBUG(...) RAFT_LOG_DEBUG("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_INFO(...) RAFT_LOG_INFO("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_WARN(...) RAFT_LOG_WARN("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_ERROR(...) RAFT_LOG_ERROR("%s", ::fmt::format(__VA_ARGS__).c_str())

#else

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace arachne {

/// Standalone fallback used only when RAFT isn't linked. Mirrors
/// raft::default_logger(): one process-wide, lazily-created logger.
inline spdlog::logger& default_logger() {
	static std::shared_ptr<spdlog::logger> logger = spdlog::stderr_color_mt("arachne");
	return *logger;
}

}  // namespace arachne

#define ARACHNE_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(&::arachne::default_logger(), __VA_ARGS__)
#define ARACHNE_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(&::arachne::default_logger(), __VA_ARGS__)
#define ARACHNE_LOG_INFO(...) SPDLOG_LOGGER_INFO(&::arachne::default_logger(), __VA_ARGS__)
#define ARACHNE_LOG_WARN(...) SPDLOG_LOGGER_WARN(&::arachne::default_logger(), __VA_ARGS__)
#define ARACHNE_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(&::arachne::default_logger(), __VA_ARGS__)

#endif
