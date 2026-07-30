#pragma once

#include <atomic>
#include <cstdint>
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
/// means beyond that: no Region dependency/write-lease bookkeeping, no
/// eviction policy. That all lives in Core's RegionManager instead. RoutingCache
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
	/// it was registered via ensure()) doesn't cover `query`. Not itself
	/// virtual -- see nearestImpl() below for the hook a concrete
	/// implementation actually overrides; this wrapper's only job is
	/// recording the hit/miss (see stats()) uniformly across every
	/// implementation, current or future, so a concrete RoutingCache never
	/// has to remember to do that bookkeeping itself.
	std::optional<VectorId> nearest(const VectorView& query) {
		std::optional<VectorId> result = nearestImpl(query);
		if (result.has_value()) {
			hits_.fetch_add(1, std::memory_order_relaxed);
		} else {
			misses_.fetch_add(1, std::memory_order_relaxed);
		}
		return result;
	}

	/// Returns the id of the entry for `vector` -- either the existing one
	/// found via the same per-anchor "close enough" test as nearest(), or a
	/// freshly registered one under `id` (with radius `max_distance`) if none
	/// exists yet.
	virtual VectorId ensure(VectorId id, const VectorView& vector, float max_distance) = 0;

	virtual void erase(VectorId id) = 0;

	/// Cumulative nearest() outcomes since construction -- hits (a candidate
	/// within its own registered radius was found) vs. misses (cache empty,
	/// or the closest candidate's radius didn't cover the query), the
	/// Anchor-routing-cache-specific counterpart to ControllerStats/
	/// RegionManager::Stats. Independent atomics (relaxed ordering): each is
	/// a running total nothing else needs to observe atomically alongside.
	struct Stats {
		std::uint64_t hits = 0;
		std::uint64_t misses = 0;
	};
	Stats stats() const {
		return Stats{hits_.load(std::memory_order_relaxed), misses_.load(std::memory_order_relaxed)};
	}

 protected:
	/// The actual per-implementation lookup -- same contract nearest() above
	/// documents (nullopt means "no candidate within its own registered
	/// max_distance"). Protected, not public: callers always go through
	/// nearest() so hit/miss accounting can never be bypassed.
	virtual std::optional<VectorId> nearestImpl(const VectorView& query) = 0;

	std::uint32_t dim_;
	DistanceMetric metric_;
	VectorDType dtype_;

 private:
	std::atomic<std::uint64_t> hits_{0};
	std::atomic<std::uint64_t> misses_{0};
};

}  // namespace arachne
