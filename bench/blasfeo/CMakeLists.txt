# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(Threads REQUIRED)

add_executable(bench-blasfeo
    Main.cpp
    Trmm.cpp
    Gemm.cpp
    Syrk.cpp
    SyrkPotrf.cpp
    Potrf.cpp
)

target_compile_definitions(bench-blasfeo
    # Use Blaze without linking to a BLAS library.
    # Blaze is used to prepare data in some of the benchmarks.
    PRIVATE BLAZE_BLAS_MODE=0
)

target_link_libraries(bench-blasfeo
    blast-blasfeo
    bench-blast-common
    ${CMAKE_THREAD_LIBS_INIT}
)
