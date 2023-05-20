# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cmake_minimum_required (VERSION 3.10)   # Need at least 3.10 for gtest_discover_tests()
project (blast)

# Enable modern C++
set(CMAKE_CXX_STANDARD 20)

# Don't use compiler's C++ extensions
set(CMAKE_CXX_EXTENSIONS False)

# Enable position-independent code on all targets by default.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include CMakeToolsHelpers for easier debugging from VS Code
include(CMakeToolsHelpers OPTIONAL)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_DEBUG_POSTFIX d)

# find_package(LAPACK REQUIRED)

# Find Blaze.
find_package(blaze REQUIRED)

# add_subdirectory(src)
# add_subdirectory(examples)

add_library(blast INTERFACE)

target_include_directories(blast
    INTERFACE ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(blast
    INTERFACE blaze::blaze
)

target_compile_options(blast
    INTERFACE "-Wno-ignored-attributes"
)

# BLAST_WITH_BLASFEO
set(BLAST_WITH_BLASFEO CACHE BOOL "Build blasfeo C++ interface")

if (BLAST_WITH_BLASFEO)
    find_package(BLASFEO REQUIRED)

    add_library(blast-blasfeo INTERFACE)

    target_include_directories(blast-blasfeo
        INTERFACE ${PROJECT_INCLUDE_DIR}
        INTERFACE ${BLASFEO_INCLUDE_DIR}
    )

    target_link_libraries(blast-blasfeo
        INTERFACE ${BLASFEO_STATIC_LIB}
        INTERFACE blast
    )
endif()

# BLAST_WITH_TEST
set(BLAST_WITH_TEST ON CACHE BOOL "Build blast tests")

if (BLAST_WITH_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# BLAST_WITH_BENCHMARK
option(BLAST_WITH_BENCHMARK "Build blast benchmarks")

if (BLAST_WITH_BENCHMARK)
    include_directories(${PROJECT_INCLUDE_DIR})
    add_subdirectory(bench)
endif()
