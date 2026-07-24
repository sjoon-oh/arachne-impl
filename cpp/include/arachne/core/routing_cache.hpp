#pragma once

#include <optional>

#include "arachne/types.hpp"

namespace arachne {

/// The "Anchor Query 기반 Semantic Routing Cache" from the Quick Summary:
/// every query passes through this first. It answers exactly one question
/// -- "is the incoming query vector close enough to a previously seen one
/// (an Anchor, identified only by its VectorId) that routing to GPU is
/// worthwhile?" -- and nothing else. It does not know what an Anchor id
/// means beyond that: no Stitch/write-lease bookkeeping, no eviction
/// policy. That all lives in Core's AnchorManager instead. RoutingCache
/// just needs very fast insert/erase, since Anchors churn constantly as
/// queries stream in; RoutingCacheHnsw (hnswlib-backed) is the first
/// concrete implementation, with Locality Sensitive Hashing remaining a
/// candidate for a leaner-delete alternative. Core is written only against
/// this interface, never against a concrete cache.
class RoutingCache {
 public:
	virtual ~RoutingCache() = default;

	/// The id of the registered entry nearest to `query`, or nullopt if none
	/// is close enough (or the cache is empty). "Close enough" is an
	/// implementation decision -- Core applies no threshold of its own.
	virtual std::optional<VectorId> nearest(const VectorView& query) = 0;

	/// Returns the id of the entry for `vector` -- either the existing one
	/// found via the same "close enough" test as nearest(), or a freshly
	/// registered one under `id` if none exists yet.
	virtual VectorId ensure(VectorId id, const VectorView& vector) = 0;

	virtual void erase(VectorId id) = 0;
};

}  // namespace arachne
