# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.10)   # Need at least 3.10 for gtest_discover_tests()
project(blast VERSION 0.1 LANGUAGES CXX)

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

find_package(Boost REQUIRED COMPONENTS exception)
find_package(blaze REQUIRED)

add_library(blast INTERFACE)

target_include_directories(blast INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(blast
    INTERFACE blaze::blaze
)

target_compile_options(blast
    INTERFACE "-Wno-ignored-attributes" "-fno-math-errno" "-ftemplate-backtrace-limit=0"
    # Enable SIMD instruction sets, otherwise it does not compile.
    # This will change when we support multiple architectures.
    INTERFACE "-march=native" "-mfma" "-mavx" "-mavx2" "-msse4"
)

# BLAST_WITH_BLASFEO
option(BLAST_WITH_BLASFEO "Build blasfeo C++ interface")

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
option(BLAST_WITH_TEST "Build blast tests")

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

#
# Install
#
install(
    DIRECTORY include/
    DESTINATION include
)

# Install targets and generate an export target
install(TARGETS blast
    EXPORT blast-targets
)

install(EXPORT blast-targets
    NAMESPACE blast::
    DESTINATION share/blast/cmake
)

# Generate and install the package configuration and version files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/blast-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/blast-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/blast-config.cmake"
    INSTALL_DESTINATION share/blast/cmake
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/blast-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/blast-config-version.cmake"
    DESTINATION share/blast/cmake
)
