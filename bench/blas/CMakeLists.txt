find_package(Threads REQUIRED)

foreach (BLAS_LIB openblas mkl_rt)
    find_library(BLAS_LIB_PATH_${BLAS_LIB} ${BLAS_LIB} 
        PATHS 
            "/opt/intel/mkl/lib/intel64_lin" "/opt/intel/mkl/lib/intel64"
            "/opt/blasfeo/lib"
    )

    if (BLAS_LIB_PATH_${BLAS_LIB})
        message(STATUS "${BLAS_LIB} found at ${BLAS_LIB_PATH_${BLAS_LIB}}. Benchmark for ${BLAS_LIB} will be built.")

        add_executable(bench-blas-${BLAS_LIB}
            Main.cpp
            Gemm.cpp
            Potrf.cpp
            Cholesky.cpp
            Syrk.cpp
            Trmm.cpp
            Trmv.cpp
            Trsv.cpp
            SyrkPotrf.cpp
        )

        target_link_libraries(bench-blas-${BLAS_LIB}
            bench-blazefeo-common
            blaze::blaze
            ${BLAS_LIB_PATH_${BLAS_LIB}}
            ${CMAKE_THREAD_LIBS_INIT}
        )
    else ()
        message(STATUS "${BLAS_LIB} not found. Benchmark for ${BLAS_LIB} will NOT be built.")
    endif ()
endforeach ()
