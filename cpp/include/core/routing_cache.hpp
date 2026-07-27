#pragma once

#include <optional>

#include "types.hpp"

namespace arachne {

/// Which distance function a RoutingCache computes closeness with. Common
/// to every concrete RoutingCache -- declared here rather than on a
/// specific implementation -- since "is this vector close enough" needs an
/// answer regardless of what data structure backs it. A concrete
/// implementation maps (metric, dtype) onto whatever its own backing index
/// needs; see ASRoutingCacheHnsw for hnswlib's version of that mapping.
///
/// Cosine is only supported with VectorDType::Float32 by ASRoutingCacheHnsw
/// today (normalizing a quantized int8/uint8 vector and re-quantizing it
/// back loses precision plain float doesn't, and Float16 would need its
/// own half-aware normalize) -- that restriction is enforced by whichever
/// concrete RoutingCache is constructed, not by this enum itself, since a
/// different implementation could in principle support more.
enum class DistanceMetric { L2, InnerProduct, Cosine };

/// The "Anchor Query 기반 Semantic Routing Cache" from the Quick Summary:
/// every query passes through this first. It answers exactly one question
/// -- "is the incoming query vector close enough to a previously seen one
/// (an Anchor, identified only by its VectorId) that routing to GPU is
/// worthwhile?" -- and nothing else. It does not know what an Anchor id
/// means beyond that: no Stitch/write-lease bookkeeping, no eviction
/// policy. That all lives in Core's AnchorManager instead. RoutingCache
/// just needs very fast insert/erase, since Anchors churn constantly as
/// queries stream in; ASRoutingCacheHnsw (hnswlib-backed) is the first
/// concrete implementation, with Locality Sensitive Hashing remaining a
/// candidate for a leaner-delete alternative. Core is written only against
/// this interface, never against a concrete cache.
///
/// "Close enough" is judged per-Anchor, not cache-wide: each Anchor carries
/// its own max_distance, fixed at the time it was registered via ensure().
/// nearest() finds the single closest candidate and accepts it only if the
/// query falls within that specific candidate's own radius -- it never
/// searches further for a different candidate with a wider radius.
///
/// dimension()/metric()/dtype() are fixed for the lifetime of a
/// RoutingCache and common to every concrete implementation -- every
/// vector passed to nearest()/ensure() is expected to carry `dimension()`
/// elements of `dtype()`. Held here (protected, set once via the
/// constructor) rather than duplicated as private state in each
/// implementation.
class RoutingCache {
 public:
	RoutingCache(std::uint32_t dim, DistanceMetric metric, VectorDType dtype)
			: dim_(dim), metric_(metric), dtype_(dtype) {}
	virtual ~RoutingCache() = default;

	std::uint32_t dimension() const { return dim_; }
	DistanceMetric metric() const { return metric_; }
	VectorDType dtype() const { return dtype_; }

	/// The id of the registered entry nearest to `query`, or nullopt if the
	/// cache is empty or the closest candidate's own max_distance (set when
	/// it was registered via ensure()) doesn't cover `query`.
	virtual std::optional<VectorId> nearest(const VectorView& query) = 0;

	/// Returns the id of the entry for `vector` -- either the existing one
	/// found via the same per-anchor "close enough" test as nearest(), or a
	/// freshly registered one under `id` (with radius `max_distance`) if none
	/// exists yet.
	virtual VectorId ensure(VectorId id, const VectorView& vector, float max_distance) = 0;

	virtual void erase(VectorId id) = 0;

 protected:
	std::uint32_t dim_;
	DistanceMetric metric_;
	VectorDType dtype_;
};

}  // namespace arachne
