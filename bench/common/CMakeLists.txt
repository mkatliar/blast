# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(benchmark REQUIRED)

add_library(bench-blazefeo-common STATIC
    Benchmark.cpp
)

target_link_libraries(bench-blazefeo-common
    PUBLIC benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)


target_compile_options(bench-blazefeo-common
    PUBLIC "-mllvm" "-inline-threshold=1000"
)