find_package(GTest CONFIG REQUIRED)

set(ARACHNE_UNITTEST_SOURCES
	routing_cache_hnsw_test.cpp
	routing_cache_hnsw_dtype_test.cpp
	region_manager_test.cpp
	region_manager_coordinator_test.cpp
	op_scheduler_policy_test.cpp
	op_scheduler_worker_start_test.cpp
	replacement_policy_test.cpp
	gpu/dirty_header_test.cpp
	util/distance_test.cpp
	gpu/unit_pool_arena_test.cpp
	gpu/compaction_policy_test.cpp
	gpu/device_region_pool_test.cpp
	controller_test.cpp
	# StressIndex itself (test/stress, shared with bin/ -- see test/CMakeLists.txt)
	# plus the gtest-driven stress stages that exercise it.
	../stress/stress_index.cpp
	stress/stress_index_test.cpp
	stress/stress_index_stage2_test.cpp
	stress/stress_index_stage3_test.cpp
)

add_executable(arachne_tests ${ARACHNE_UNITTEST_SOURCES})

target_include_directories(arachne_tests PRIVATE ../stress)

target_link_libraries(arachne_tests PRIVATE
	arachne::core
	GTest::gtest
	GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(arachne_tests)
