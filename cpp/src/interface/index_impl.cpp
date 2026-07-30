#include "interface/index_impl.hpp"

#include <utility>

namespace arachne {

IndexImpl::IndexImpl(std::unique_ptr<IAdapter> adapter,
							 std::unique_ptr<RoutingCache> routing_cache,
							 const SchedulingConfig& scheduling_config)
	: adapter_(std::move(adapter)),
		routing_cache_(std::move(routing_cache)),
		controller_(*adapter_, *routing_cache_, scheduling_config) {}

SearchResult IndexImpl::search(const Query& query) { return controller_.search(query); }

InsertResult IndexImpl::insert(const Record& record) { return controller_.insert(record); }

DeleteResult IndexImpl::remove(VectorId id) { return controller_.remove(id); }

}  // namespace arachne
