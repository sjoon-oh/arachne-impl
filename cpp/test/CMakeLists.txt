find_package(GTest CONFIG REQUIRED)

add_executable(arachne_tests
	routing_cache_hnsw_test.cpp
	anchor_manager_test.cpp
)

target_link_libraries(arachne_tests PRIVATE
	arachne::core
	GTest::gtest
	GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(arachne_tests)
