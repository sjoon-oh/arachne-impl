# Standalone, plain-main() applications -- not gtest, not run via ctest.
# Each is meant to be invoked directly (e.g. ./arachne_full_suite_app --help)
# to eyeball a real Controller-driven workload's behavior/timing.

add_executable(arachne_full_suite_app
	full_suite_app.cpp
	../stress/stress_index.cpp
)

target_include_directories(arachne_full_suite_app PRIVATE ../stress)

target_link_libraries(arachne_full_suite_app PRIVATE arachne::core)
