#pragma once

#include <memory>

#include "adapter/index_adapter.hpp"
#include "core/controller.hpp"
#include "core/routing_cache.hpp"
#include "interface/index.hpp"

namespace arachne {

/// The default Index implementation, and the only class an application
/// talks to today: exposes search/insert/delete and forwards everything to
/// Controller, which owns the actual Arachne control plane. IndexImpl itself
/// carries no policy -- it just owns the adapter and routing cache and the
/// Controller wired to them.
class IndexImpl : public Index {
 public:
	IndexImpl(std::unique_ptr<IAdapter> adapter, std::unique_ptr<RoutingCache> routing_cache,
				 const SchedulingConfig& scheduling_config = {});

	SearchResult search(const Query& query) override;
	InsertResult insert(const Record& record) override;
	DeleteResult remove(VectorId id) override;

 private:
	std::unique_ptr<IAdapter> adapter_;
	std::unique_ptr<RoutingCache> routing_cache_;
	Controller controller_;
};

}  // namespace arachne
