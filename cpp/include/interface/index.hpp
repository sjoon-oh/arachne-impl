#pragma once

#include "types.hpp"

namespace arachne {

/// Abstract entry point applications program against. Mirrors the
/// RoutingCache pattern one layer down: RoutingCache lets Core stay
/// implementation-agnostic about the routing cache; Index lets application
/// code stay implementation-agnostic about which concrete top-level
/// implementation actually serves SEARCH/INSERT/DELETE. Engine is the
/// first implementation, built on Core; other implementations (a different
/// control-plane strategy, a test double, ...) can plug in alongside it
/// without application code changing.
class Index {
 public:
	virtual ~Index() = default;

	virtual SearchResult search(const Query& query) = 0;
	virtual InsertResult insert(const Record& record) = 0;
	virtual DeleteResult remove(VectorId id) = 0;
};

}  // namespace arachne
