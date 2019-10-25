cmake_minimum_required (VERSION 3.10)   # Need at least 3.10 for gtest_discover_tests()
project (smoke)

# Enable modern C++
set(CMAKE_CXX_STANDARD 17)

# Don't use compiler's C++ extensions
set(CMAKE_CXX_EXTENSIONS False)

# Enable position-independent code on all targets by default.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include CMakeToolsHelpers for easier debugging from VS Code
include(CMakeToolsHelpers OPTIONAL)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_DEBUG_POSTFIX d)

# find_package(LAPACK REQUIRED)

# Find boost.
find_package(Boost REQUIRED COMPONENTS exception)

# Find Blaze.
find_package(blaze REQUIRED)

# add_subdirectory(src)
# add_subdirectory(examples)

add_library(smoke INTERFACE)

target_include_directories(smoke
    INTERFACE ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(smoke
    INTERFACE blaze::blaze
)

# SMOKE_WITH_TEST
set(SMOKE_WITH_TEST ON CACHE BOOL "Build smoke tests")

if (SMOKE_WITH_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# SMOKE_WITH_BENCHMARK
set(SMOKE_WITH_BENCHMARK ON CACHE BOOL "Build smoke benchmarks")

if (SMOKE_WITH_BENCHMARK)
    add_subdirectory(bench)
endif()
