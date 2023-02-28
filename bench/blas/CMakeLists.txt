# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(Threads REQUIRED)

foreach (BLA_VENDOR OpenBLAS Intel10_64lp_seq Intel10_64_dyn)
    find_package(BLAS)

    if (BLAS_FOUND)
        message(STATUS "${BLA_VENDOR} libraries found. Benchmark for ${BLA_VENDOR} will be built.")

        add_executable(bench-blas-${BLA_VENDOR}
            Main.cpp
            Gemm.cpp
            Getrf.cpp
            Potrf.cpp
            Cholesky.cpp
            Syrk.cpp
            Trmm.cpp
            Trmv.cpp
            Trsv.cpp
            SyrkPotrf.cpp
        )

        target_link_libraries(bench-blas-${BLA_VENDOR}
            bench-blazefeo-common
            blaze::blaze
            ${BLAS_LIBRARIES}
        )

        target_link_options(bench-blas-${BLA_VENDOR}
            PRIVATE ${BLAS_LINKER_FLAGS}
        )
    else ()
        message(STATUS "${BLA_VENDOR} not found. Benchmark for ${BLA_VENDOR} will NOT be built.")
    endif ()
endforeach ()
