# Two consumers of test/stress's shared StressIndex adapter (see its own
# class doc comment): unittest/ (gtest-driven, run via ctest) and bin/ (a
# plain main()-having executable you run directly, for eyeballing a full
# insert/search/remove workload against a real Controller without going
# through gtest at all). StressIndex itself lives once, here, under
# test/stress -- neither subdirectory owns it.
add_subdirectory(unittest)
add_subdirectory(bin)
# index/: a real-dataset (not synthetic) comparison of raw hnswlib vs the
# full Arachne stack -- see index/hnsw_workload_compare.cpp's file overview.
add_subdirectory(index)
