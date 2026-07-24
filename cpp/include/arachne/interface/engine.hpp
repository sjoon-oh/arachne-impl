#pragma once

#include <memory>

#include "arachne/adapter/index_adapter.hpp"
#include "arachne/core/core.hpp"
#include "arachne/core/routing_cache.hpp"

namespace arachne {

/// The only class an application talks to: exposes search/insert/delete and
/// forwards everything to Core, which owns the actual Arachne control
/// plane. Engine itself carries no policy -- it just owns the adapter and
/// routing cache and the Core wired to them.
class Engine {
 public:
	Engine(std::unique_ptr<IndexAdapter> adapter, std::unique_ptr<RoutingCache> routing_cache);

	SearchResult search(const Query& query);
	InsertResult insert(const Record& record);
	DeleteResult remove(VectorId id);

 private:
	std::unique_ptr<IndexAdapter> adapter_;
	std::unique_ptr<RoutingCache> routing_cache_;
	Core core_;
};

}  // namespace arachne
