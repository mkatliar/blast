# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(benchmark REQUIRED)

add_library(bench-blast-common STATIC
    Benchmark.cpp
    Syrk.cpp
)

target_link_libraries(bench-blast-common
    PUBLIC benchmark::benchmark
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # More aggressive inlining with Clang
    target_compile_options(bench-blast-common
        PUBLIC "-mllvm" "-inline-threshold=16000"
    )
endif()
