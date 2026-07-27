cmake_minimum_required(VERSION 3.26)

project(arachne LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Blackwell: SM100 (B100/B200/GB200 data-center) + SM120 (RTX 50-series /
# GeForce Blackwell). Override with -DCMAKE_CUDA_ARCHITECTURES=... for other
# targets (e.g. Hopper=90).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 100 120)
endif()

option(ARACHNE_USE_RAFT "Link against RAFT (installed via the conda env) for GPU primitives" ON)
option(ARACHNE_BUILD_TESTS "Build unit tests (requires GTest, see conda/arachne-blackwell.yml)" ON)

# Resolve conda-provided dependencies (raft, ...) from the active env.
if(DEFINED ENV{CONDA_PREFIX})
	list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# Header-only; vendored as a submodule rather than pulled from conda since
# it isn't packaged there. Backs ASRoutingCacheHnsw, but only inside
# routing_cache_hnsw.cpp -- the public header (arachne/core/
# routing_cache_hnsw.hpp) forward-declares its hnswlib-wrapping Instance
# type, so hnswlib stays a PRIVATE (impl-only) dependency of arachne_core.
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib SYSTEM INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/hnswlib>
)

set(ARACHNE_CORE_SOURCES
		src/interface/index_impl.cpp
		src/core/controller.cpp
		src/core/op_scheduler.cpp
		src/core/scheduling_policy.cpp
		src/core/anchor_manager.cpp
		src/core/as_routing_cache.cpp
	src/core/as_routing_cache_hnsw.cpp
	src/util/distance.cpp
)
if(ARACHNE_USE_RAFT)
	# GPU device/memory management (arachne/gpu) -- built on raft::device_resources
	# and RMM, so it only compiles/links when RAFT is actually available. See
	# gpu/device_context.hpp and gpu/stitch_pool.hpp; Controller only declares
	# its gpu:: members under the matching ARACHNE_WITH_RAFT compile definition
	# set below.
	list(APPEND ARACHNE_CORE_SOURCES
		src/gpu/device_context.cpp
		src/gpu/stitch_pool.cpp
	)
endif()

add_library(arachne_core ${ARACHNE_CORE_SOURCES})
add_library(arachne::core ALIAS arachne_core)

target_include_directories(arachne_core PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# PRIVATE, cpp/ itself (not cpp/include): src/util/distance.cpp's Highway
# foreach_target.h self-include re-includes this file by its src/-relative
# path (see HWY_TARGET_INCLUDE there), which needs cpp/ -- not cpp/include,
# which distance.cpp isn't under -- on the include path to resolve.
target_include_directories(arachne_core PRIVATE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_compile_features(arachne_core PUBLIC cxx_std_20)
target_link_libraries(arachne_core PRIVATE hnswlib)

# logging.hpp needs fmt::format directly (see its own comment) regardless of
# which ARACHNE_USE_RAFT branch is active -- the spdlog fallback pulls fmt
# in transitively, but raft::raft does not, so this can't be left implicit.
find_package(fmt CONFIG REQUIRED)
target_link_libraries(arachne_core PUBLIC fmt::fmt)

# Highway: CPU SIMD (AVX2/AVX512/NEON) with runtime target dispatch, backing
# arachne/util/distance.hpp. Impl-only (src/util/distance.cpp), same as hnswlib --
# the public header exposes plain float* signatures, so nothing that only
# includes it needs Highway on its include path.
find_package(hwy CONFIG REQUIRED)
target_link_libraries(arachne_core PRIVATE hwy::hwy)

# ASRoutingCacheHnsw's background compaction (std::thread/std::shared_mutex).
find_package(Threads REQUIRED)
target_link_libraries(arachne_core PUBLIC Threads::Threads)

if(ARACHNE_USE_RAFT)
	find_package(raft CONFIG REQUIRED)
	target_link_libraries(arachne_core PUBLIC raft::raft)
	target_compile_definitions(arachne_core PUBLIC ARACHNE_WITH_RAFT)
else()
	# Logging fallback when RAFT (and its bundled rapids-logger/spdlog) isn't
	# linked -- see cpp/include/arachne/logging.hpp.
	find_package(spdlog CONFIG REQUIRED)
	target_link_libraries(arachne_core PUBLIC spdlog::spdlog)
endif()

if(ARACHNE_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
