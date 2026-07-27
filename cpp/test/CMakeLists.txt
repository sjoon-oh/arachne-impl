find_package(GTest CONFIG REQUIRED)

set(ARACHNE_TEST_SOURCES
	routing_cache_hnsw_test.cpp
	routing_cache_hnsw_dtype_test.cpp
	anchor_manager_test.cpp
	op_scheduler_policy_test.cpp
	util/distance_test.cpp
)
if(ARACHNE_USE_RAFT)
	list(APPEND ARACHNE_TEST_SOURCES gpu/stitch_pool_test.cpp)
endif()

add_executable(arachne_tests ${ARACHNE_TEST_SOURCES})

target_link_libraries(arachne_tests PRIVATE
	arachne::core
	GTest::gtest
	GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(arachne_tests)
