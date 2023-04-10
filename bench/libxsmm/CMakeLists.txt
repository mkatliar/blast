# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(benchmark REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(BLAS REQUIRED)

pkg_check_modules(libxsmm REQUIRED IMPORTED_TARGET libxsmm)

add_executable(bench-libxsmm
    Main.cpp
    Gemm.cpp
)

target_link_libraries(bench-libxsmm
    blast
    benchmark::benchmark
    BLAS::BLAS
    PkgConfig::libxsmm
)
