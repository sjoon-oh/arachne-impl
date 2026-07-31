#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/routing_cache.hpp"

namespace arachne {

/// RoutingCache implemented as an Active/Shadow pair with background,
/// tombstone-ratio-triggered compaction. This is the concurrency and
/// lifecycle machinery any soft-delete-based ANN index integration needs
/// (ASRoutingCacheHnsw today; the long-standing LSH candidate mentioned in
/// RoutingCache's own docs would be another) -- factored out here so a
/// concrete integration only has to implement RefreshManager (one index's worth
/// of insert/erase/search/snapshot, including whatever incremental
/// add/build story that index has -- entirely that implementation's own
/// business) and a factory that builds a fresh empty one. *When* to
/// rebuild, how to snapshot without blocking readers, and how to swap in
/// the result live here exactly once, index-agnostically.
///
/// Concurrency: a RefreshManager's own locking (if any) is not assumed to
/// protect concurrent reads against concurrent writes -- confirmed false
/// for hnswlib by reading its source (searchKnn walks link lists without
/// taking any of hnswlib's own locks), and not something a future
/// integration should have to re-verify per index library. This class
/// compensates with its own shared_mutex around every access to the active
/// RefreshManager: nearest() takes a shared (read) lock, ensure()/erase() take an
/// exclusive (write) lock. A RefreshManager is never touched outside one of
/// those two lock modes.
///
/// Deletion: a RefreshManager is assumed to only tombstone (mark dead, not
/// reclaim space) on erase() -- see RefreshManager::tombstoneCount(). Once the
/// active RefreshManager's tombstone ratio crosses max_tombstone_ratio, a
/// background thread builds a fresh "shadow" RefreshManager (via makeRefreshManager())
/// from the active one's live (id, vector, max_distance) triples and swaps
/// to it once ready. The rebuild itself holds no lock (readers and writers
/// of the active RefreshManager proceed normally throughout); only a brief
/// initial snapshot and a final reconcile-and-swap pass (bounded by how
/// much changed during the rebuild, not by N) take the shared/exclusive
/// lock respectively.
///
/// Eviction policy -- which ids get erase()'d in the first place -- is
/// entirely the caller's (Core's) call; this class only ever reclaims
/// space already marked dead.
///
/// Lifecycle: every subclass MUST call waitForCompaction() as the first
/// statement in its own destructor, before any subclass members are
/// destroyed. The background compaction thread calls makeRefreshManager(),
/// a pure virtual only the subclass implements; C++ resets an object's
/// dynamic type for virtual dispatch as each destructor runs, so once the
/// subclass's own destructor has started (or finished, if it doesn't
/// declare one) the override is no longer reachable, and a still-running
/// compaction thread calling makeRefreshManager() crashes with "pure
/// virtual method called". ~ASRoutingCache() also joins the thread, but by
/// then it's too late for this reason -- that's defense in depth for the
/// case it already finished on its own, not a substitute.
class ASRoutingCache : public RoutingCache {
 public:
	/// Index-agnostic callback surface one Active or Shadow RefreshManager exposes
	/// to ASRoutingCache. Deliberately operates on raw `const void*`
	/// (matching most ANN libraries' own APIs, hnswlib included) rather than
	/// VectorView: which VectorDType those bytes are is fixed for the whole
	/// ASRoutingCache (RoutingCache::dtype()) and checked once at the
	/// nearest()/ensure() boundary, not per-RefreshManager-call. Everything about
	/// *how* the underlying index builds itself up incrementally --
	/// insert/erase/search, its own data layout -- is this interface
	/// implementation's business alone; ASRoutingCache never looks inside
	/// it, only drives it through this surface plus makeRefreshManager() below.
	class RefreshManager {
	 public:
		virtual ~RefreshManager() = default;

		virtual void insert(VectorId id, const void* vector_data, float max_distance) = 0;
		virtual bool erase(VectorId id) = 0;

		/// Index-nearest by whatever the underlying index's own notion of
		/// closest is, accepted by ASRoutingCache only if the query falls
		/// within *that candidate's own* registered max_distance -- see
		/// RoutingCache's class docs. Never widens the search to consider a
		/// farther candidate with a larger radius.
		virtual std::optional<VectorId> findNearest(const void* query_data) = 0;

		/// Raw bytes of the vector stored for `id` (RoutingCache::dimension()
		/// * VectorElementSize(RoutingCache::dtype()) of them) -- used during
		/// compaction to carry an id inserted on the active RefreshManager
		/// mid-rebuild into the shadow.
		virtual std::vector<std::byte> rawVectorOf(VectorId id) = 0;

		virtual float maxDistanceOf(VectorId id) const = 0;

		/// Safe to expose by reference (not copy): every caller (compaction's
		/// reconcile phase) uses it synchronously, under the same lock that
		/// already protects this RefreshManager, without letting the reference
		/// escape.
		virtual const std::unordered_set<VectorId>& liveIds() const = 0;

		virtual std::size_t liveCount() const = 0;
		virtual std::size_t tombstoneCount() const = 0;

		/// Visits every surviving id, its raw vector bytes, and its own
		/// max_distance -- used during compaction to snapshot live state
		/// before migrating it into a fresh RefreshManager.
		virtual void forEachLive(const std::function<void(VectorId, const void*, float)>& fn) const = 0;
	};

	~ASRoutingCache() override;

	VectorId ensure(VectorId id, const VectorView& vector, float max_distance) override;
	void erase(VectorId id) override;

	/// Blocks until any in-flight compaction finishes. Not part of the
	/// RoutingCache interface -- normal operation never needs this. Exists for
	/// tests that want to observe post-compaction state deterministically, and
	/// as the subclass-destructor synchronization point the class doc above
	/// requires.
	void waitForCompaction();

 protected:
	/// `initial_active` is built by the subclass itself (via an ordinary,
	/// non-virtual call to whatever makeRefreshManager() will later delegate
	/// to), not by this constructor calling makeRefreshManager() -- a virtual
	/// call from a base constructor can't reach a derived override anyway,
	/// since the derived part of the object doesn't exist yet.
	ASRoutingCache(std::uint32_t dim, DistanceMetric metric, VectorDType dtype,
								 std::size_t initial_capacity, double max_tombstone_ratio,
								 std::unique_ptr<RefreshManager> initial_active);

	/// Builds one fresh, empty RefreshManager with room for `capacity` entries
	/// -- the one ANN-index-specific operation this class can't implement
	/// itself. Called only from compactImpl() to build the shadow, never from
	/// the constructor (see `initial_active` above for why).
	virtual std::unique_ptr<RefreshManager> makeRefreshManager(std::size_t capacity) const = 0;

 private:
	std::optional<VectorId> nearestImpl(const VectorView& query) override;

	/// Launches compactImpl() on a background thread if one isn't already
	/// running. Reaps (joins) the previous thread first; a fresh
	/// compare_exchange success implies it already finished (compacting_ only
	/// clears at the very end of compactImpl()).
	void triggerCompaction();

	/// Runs on a background thread. Snapshots active_'s live
	/// (id, vector, max_distance) triples under a brief shared lock, rebuilds
	/// a fresh RefreshManager (via makeRefreshManager()) from that snapshot with no lock
	/// held, then reconciles whatever inserted/erased on active_ meanwhile
	/// and swaps in, under a final brief exclusive lock.
	void compactImpl();

	std::size_t initial_capacity_;
	double max_tombstone_ratio_;

	mutable std::shared_mutex mutex_;
	std::unique_ptr<RefreshManager> active_;

	std::atomic<bool> compacting_{false};
	std::thread compaction_thread_;
};

}  // namespace arachne
