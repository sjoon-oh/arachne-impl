# HnswIndex: the real (not test-double) IAdapter over thirdparty/hnswlib.
# See hnsw_index.hpp for its [SKELETON] status and report.md for the design.
#
# A separate library from arachne_core (rather than folded into
# ARACHNE_CORE_SOURCES) because it's an adapter -- a *consumer* of
# arachne_core's IAdapter interface, not part of the control-plane core
# itself, same relationship test/stress/stress_index.hpp has to arachne_core
# (see ../../test/CMakeLists.txt). hnswlib itself was already made available
# as a plain INTERFACE target (include-path only) by the top-level
# CMakeLists.txt.
add_library(arachne_index_hnsw
	hnsw_index.cpp
)
add_library(arachne::index_hnsw ALIAS arachne_index_hnsw)

target_link_libraries(arachne_index_hnsw
	PUBLIC arachne::core
	PRIVATE hnswlib
)
