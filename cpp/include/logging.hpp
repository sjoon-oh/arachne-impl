#pragma once

// Arachne's logging facade: one set of macros (ARACHNE_LOG_*) used
// throughout include/src, forwarding to RAFT's own logger
// (raft::default_logger(), a rapids-logger/spdlog wrapper) so Arachne's
// control-plane logs and RAFT's GPU-primitive logs share one sink, pattern,
// and level -- RAFT is a hard dependency of arachne_core (see
// CMakeLists.txt), so there is no non-RAFT fallback to select between.
//
// spdlog is the engine underneath, but rapids_logger::logger::log() (what
// RAFT_LOG_* forwards to) formats with plain printf-style specifiers
// (std::snprintf internally) and silently drops "{}" as inert literal text
// instead of substituting it -- every ARACHNE_LOG_* call site in this
// codebase is written fmt-style, so the message is formatted eagerly with
// fmt itself and handed to raft's logger already-formatted, via "%s".

#include <fmt/format.h>
#include <raft/core/logger.hpp>

#define ARACHNE_LOG_TRACE(...) RAFT_LOG_TRACE("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_DEBUG(...) RAFT_LOG_DEBUG("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_INFO(...) RAFT_LOG_INFO("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_WARN(...) RAFT_LOG_WARN("%s", ::fmt::format(__VA_ARGS__).c_str())
#define ARACHNE_LOG_ERROR(...) RAFT_LOG_ERROR("%s", ::fmt::format(__VA_ARGS__).c_str())
