#include "arachne/interface/engine.hpp"

#include <utility>

namespace arachne {

Engine::Engine(std::unique_ptr<IndexAdapter> adapter, std::unique_ptr<RoutingCache> routing_cache)
	: adapter_(std::move(adapter)),
		routing_cache_(std::move(routing_cache)),
		core_(*adapter_, *routing_cache_) {}

SearchResult Engine::search(const Query& query) { return core_.search(query); }

InsertResult Engine::insert(const Record& record) { return core_.insert(record); }

DeleteResult Engine::remove(VectorId id) { return core_.remove(id); }

}  // namespace arachne
