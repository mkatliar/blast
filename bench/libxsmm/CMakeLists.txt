# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(libxsmm REQUIRED IMPORTED_TARGET libxsmm)
pkg_check_modules(openblas REQUIRED IMPORTED_TARGET openblas)

add_executable(bench-libxsmm
    Main.cpp
    Gemm.cpp
)

target_link_libraries(bench-libxsmm
    blazefeo
    benchmark
    PkgConfig::openblas
    PkgConfig::libxsmm
    ${CMAKE_THREAD_LIBS_INIT}
)
