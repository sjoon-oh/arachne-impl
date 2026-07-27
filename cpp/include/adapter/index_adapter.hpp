#pragma once

#include <vector>

#include "adapter/region.hpp"
#include "types.hpp"

namespace arachne {

enum class ExecutionMode {
	GpuOnly,  // must complete entirely within the given scope's resident regions
	Hybrid,   // the underlying index may fall back to Host traversal
};

struct TraverseRequest {
	Query query;
	ExecutionMode mode = ExecutionMode::Hybrid;
	RegionFootprint scope;  // regions the traversal is confined to when GpuOnly
};

struct TraverseResult {
	SearchResult result;
	RegionFootprint touched;               // footprint actually accessed
	bool completed_within_scope = false;   // false => a GpuOnly attempt fell short
};

enum class ModifyOp { Insert, Delete };

struct ModifyRequest {
	ModifyOp op = ModifyOp::Insert;
	Record record;           // valid for Insert
	VectorId target = 0;     // valid for Delete
	ExecutionMode mode = ExecutionMode::Hybrid;
	RegionFootprint scope;
	LeaseHandle lease;  // set when a GPU write lease already covers `scope`
};

struct ModifyResult {
	bool ok = false;
	RegionFootprint touched;   // regions the candidate search/traversal read
	RegionFootprint modified;  // regions actually mutated
};

/// The only surface an underlying ANNS index must implement to be driven by
/// Arachne. Per Quick Summary: SEARCH = Traversal, INSERT = Traversal ->
/// Modification, DELETE = Modification. Arachne's Core owns *where*
/// traverse/modify run (Host/GPU/hybrid, which regions); the index owns
/// *how*. Implementing this (plus IRegion) is the integration point for a
/// concrete index -- left for a future, separate piece of work.
class IndexAdapter {
 public:
	virtual ~IndexAdapter() = default;

	virtual TraverseResult traverse(const TraverseRequest& request) = 0;
	virtual ModifyResult modify(const ModifyRequest& request) = 0;

	/// Structural accessors, not primitives: let Core resolve footprints
	/// returned above into IRegion callbacks for residency/lease management.
	virtual IRegion* resolveRegion(RegionId id) = 0;
	virtual std::vector<RegionId> allRegions() const = 0;
};

}  // namespace arachne
