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
// spdlog is the engine underneath either way; RAFT just wraps it behind
// rapids-logger's PImpl, which is why the two branches below look
// different. Both accept fmt-style "{}" placeholders.

#if defined(ARACHNE_WITH_RAFT)

#include <raft/core/logger.hpp>

#define ARACHNE_LOG_TRACE(...) RAFT_LOG_TRACE(__VA_ARGS__)
#define ARACHNE_LOG_DEBUG(...) RAFT_LOG_DEBUG(__VA_ARGS__)
#define ARACHNE_LOG_INFO(...) RAFT_LOG_INFO(__VA_ARGS__)
#define ARACHNE_LOG_WARN(...) RAFT_LOG_WARN(__VA_ARGS__)
#define ARACHNE_LOG_ERROR(...) RAFT_LOG_ERROR(__VA_ARGS__)

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
