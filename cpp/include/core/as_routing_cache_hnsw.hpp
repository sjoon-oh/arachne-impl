#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "core/as_routing_cache.hpp"

namespace arachne {

/// RoutingCache backed by hnswlib's HierarchicalNSW graph. All of the
/// concurrency/compaction machinery (locking, background rebuild, Active/
/// Shadow swap) lives in ASRoutingCache; this class only knows how to
/// build one hnswlib-backed ASRoutingCache::RefreshManager for a given
/// (dimension(), metric(), dtype()) -- see makeRefreshManager() below and
/// makeHnswRefreshManager() in routing_cache_hnsw.cpp, which does the actual
/// (VectorDType, DistanceMetric) -> concrete hnswlib Space selection
/// (L2Space/L2SpaceHalf/L2SpaceI/L2SpaceInt8/InnerProductSpace/
/// InnerProductSpaceHalf/InnerProductSpaceU8/InnerProductSpaceInt8, or the
/// Cosine-over-InnerProduct special case).
///
/// Cosine (see DistanceMetric in routing_cache.hpp) is only supported here
/// with VectorDType::Float32: normalizing an int8/uint8-quantized vector
/// and re-quantizing it back loses precision in a way plain float doesn't,
/// and Float16's normalize path isn't implemented either (would need a
/// half-aware normalize, not just reusing arachne::util::Normalize as-is).
/// Constructing this class with Cosine and a non-Float32 dtype throws.
///
/// max_distance's units depend on `metric`: squared L2 distance for
/// DistanceMetric::L2, `1 - dot(x, y)` for InnerProduct/Cosine (Cosine's
/// inputs are normalized first, so its range is bounded to [0, 2];
/// InnerProduct's is not).
///
/// hnswlib is a .cpp-only dependency: nothing here names an hnswlib type,
/// so nothing that only includes this header needs hnswlib on its include
/// path.
class ASRoutingCacheHnsw : public ASRoutingCache {
 public:
	/// `max_tombstone_ratio` is the deleted/live fraction that triggers a
	/// compaction swap (see ASRoutingCache). `M`/`ef_construction` are
	/// hnswlib's own HNSW graph hyperparameters -- untuned placeholders,
	/// like the rest of the defaults here. `metric` and `dtype` are both
	/// fixed for the lifetime of the cache -- every vector passed to
	/// ensure()/nearest() is expected to carry this same VectorView::dtype
	/// (checked; mismatches throw).
	explicit ASRoutingCacheHnsw(std::uint32_t dim, std::size_t initial_capacity = 1024,
														 double max_tombstone_ratio = 0.2, std::size_t M = 16,
														 std::size_t ef_construction = 200,
														 DistanceMetric metric = DistanceMetric::L2,
														 VectorDType dtype = VectorDType::Float32);

	/// Required override of ASRoutingCache's implicit contract: waits for
	/// any in-flight compaction (which calls this class's makeRefreshManager()
	/// override) before any of this class's own members are destroyed --
	/// see waitForCompaction()'s doc comment for why that ordering matters.
	~ASRoutingCacheHnsw() override;

 protected:
	/// hnswlib-specific implementation of ASRoutingCache's one required
	/// hook: build a fresh, empty RefreshManager with room for `capacity` entries.
	std::unique_ptr<RefreshManager> makeRefreshManager(std::size_t capacity) const override;

 private:
	std::size_t M_;
	std::size_t ef_construction_;
};

}  // namespace arachne
