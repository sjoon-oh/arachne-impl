#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>

#include "arachne/core/routing_cache.hpp"

namespace arachne {

/// RoutingCache backed by hnswlib's HierarchicalNSW graph, safe for
/// concurrent callers and with a real delete story bolted on top.
///
/// Concurrency: hnswlib's own locking does not protect concurrent reads
/// (searchKnn/getDataByLabel) against concurrent writes
/// (addPoint/markDelete/resizeIndex) -- confirmed by reading hnswlib's
/// source (searchKnn walks link lists without taking any of hnswlib's
/// locks) and matches hnswlib's own documented restriction. This class
/// compensates with its own shared_mutex around every access to the active
/// Instance: nearest() takes a shared (read) lock, ensure()/erase() take an
/// exclusive (write) lock. hnswlib is never touched outside one of those
/// two lock modes.
///
/// Deletion: hnswlib itself only tombstones (markDelete) -- no real space
/// reclamation. Once the active Instance's tombstone ratio crosses
/// max_tombstone_ratio, a background thread builds a fresh "shadow"
/// Instance from the active one's live (id, vector) pairs and swaps to it
/// once ready. Unlike before Stitches moved out to Core's AnchorManager,
/// there is no per-id state left to preserve across that rebuild -- a
/// (id, vector) pair is all compaction ever has to carry over. The rebuild
/// itself holds no lock (readers and writers of the active Instance proceed
/// normally throughout); only a brief initial snapshot and a final
/// reconcile-and-swap pass (bounded by how much changed during the
/// rebuild, not by N) take the shared/exclusive lock respectively.
///
/// Eviction policy -- which ids get erase()'d in the first place -- is
/// entirely Core's call; this class only ever reclaims space already
/// marked dead.
///
/// hnswlib is a .cpp-only dependency: the Instance type wrapping one
/// hnswlib index is forward-declared here and defined in
/// routing_cache_hnsw.cpp, so nothing that only includes this header needs
/// hnswlib on its include path.
class RoutingCacheHnsw : public RoutingCache {
 public:
	/// `max_distance` is the squared L2 distance beyond which a nearest-match
	/// lookup reports no match. `max_tombstone_ratio` is the deleted/live
	/// fraction that triggers a compaction swap. All three are untuned
	/// placeholders.
	explicit RoutingCacheHnsw(std::uint32_t dim, std::size_t initial_capacity = 1024,
														 float max_distance = 1e-3f, double max_tombstone_ratio = 0.2,
														 std::size_t M = 16, std::size_t ef_construction = 200);
	~RoutingCacheHnsw() override;

	std::optional<VectorId> nearest(const VectorView& query) override;
	VectorId ensure(VectorId id, const VectorView& vector) override;
	void erase(VectorId id) override;

	/// Blocks until any in-flight compaction finishes. Not part of the
	/// RoutingCache interface -- normal operation never needs this; it exists
	/// for tests that want to observe post-compaction state deterministically
	/// and for a graceful-shutdown synchronization point.
	void waitForCompaction();

 private:
	class Instance;  // one hnswlib index + its live-id bookkeeping; defined in the .cpp

	/// Launches compactImpl() on a background thread if one isn't already
	/// running. Reaps (joins) the previous compaction thread first; by the
	/// time a new one is triggered, the old one has necessarily already
	/// finished its work (compacting_ only clears at the very end of
	/// compactImpl(), so a fresh compare_exchange success implies that).
	void triggerCompaction();

	/// Runs on a background thread. Snapshots active_'s live (id, vector)
	/// pairs under a brief shared lock, rebuilds a fresh Instance from that
	/// snapshot with no lock held, then reconciles whatever inserted/erased
	/// on active_ meanwhile and swaps in, under a final brief exclusive lock.
	void compactImpl();

	std::uint32_t dim_;
	std::size_t initial_capacity_;
	std::size_t M_;
	std::size_t ef_construction_;
	float max_distance_;
	double max_tombstone_ratio_;

	mutable std::shared_mutex mutex_;
	std::unique_ptr<Instance> active_;

	std::atomic<bool> compacting_{false};
	std::thread compaction_thread_;
};

}  // namespace arachne
