# arachne_hnsw_workload_compare: a plain main()-having, non-gtest executable
# (same spirit as ../bin's arachne_full_suite_app) that replays a real,
# pre-generated streaming ANN workload against raw hnswlib and the full
# Arachne Controller stack side by side. See hnsw_workload_compare.cpp's
# file overview and run_compare.sh for how to point it at a dataset.
add_executable(arachne_hnsw_workload_compare
	hnsw_workload_compare.cpp
	workload_dataset.cpp
)

target_include_directories(arachne_hnsw_workload_compare PRIVATE ../../index/hnsw)

target_link_libraries(arachne_hnsw_workload_compare PRIVATE
	arachne::core
	arachne::index_hnsw
	hnswlib
)

# build_and_export_index: build-only-and-export tooling for pre-building
# large-scale graphs ahead of time -- see its own file overview.
add_executable(arachne_build_and_export_index
	build_and_export_index.cpp
	workload_dataset.cpp
)

target_include_directories(arachne_build_and_export_index PRIVATE ../../index/hnsw)

target_link_libraries(arachne_build_and_export_index PRIVATE
	arachne::core
	arachne::index_hnsw
	hnswlib
)
