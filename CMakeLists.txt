cmake_minimum_required (VERSION 3.10)   # Need at least 3.10 for gtest_discover_tests()
project (blazefeo)

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

# Find boost.
find_package(Boost REQUIRED COMPONENTS exception)

# Find Blaze.
find_package(blaze REQUIRED)

# add_subdirectory(src)
# add_subdirectory(examples)

add_library(blazefeo INTERFACE)

target_include_directories(blazefeo
    INTERFACE ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(blazefeo
    INTERFACE blaze::blaze
)

# BLAZEFEO_WITH_BLASFEO
set(BLAZEFEO_WITH_BLASFEO ON CACHE BOOL "Build blasfeo C++ interface")

if (BLAZEFEO_WITH_BLASFEO)
    find_package(BLASFEO REQUIRED)

    add_library(blazefeo-blasfeo INTERFACE)

    target_include_directories(blazefeo-blasfeo
        INTERFACE ${PROJECT_INCLUDE_DIR}
        INTERFACE ${BLASFEO_INCLUDE_DIR}
    )

    target_link_libraries(blazefeo-blasfeo
        INTERFACE ${BLASFEO_STATIC_LIB}
        INTERFACE blazefeo
    )
endif()

# BLAZEFEO_WITH_TEST
set(BLAZEFEO_WITH_TEST ON CACHE BOOL "Build blazefeo tests")

if (BLAZEFEO_WITH_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# BLAZEFEO_WITH_BENCHMARK
set(BLAZEFEO_WITH_BENCHMARK ON CACHE BOOL "Build blazefeo benchmarks")

if (BLAZEFEO_WITH_BENCHMARK)
    include_directories(${PROJECT_INCLUDE_DIR})
    add_subdirectory(bench)
endif()
