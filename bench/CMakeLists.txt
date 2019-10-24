find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-smoke
    Main.cpp
    
    gemm_kernel/Gemm.cpp
    gemm_kernel/Store.cpp
    gemm_kernel/StoreVariableSize.cpp

    GemmStatic.cpp
    GemmDynamic.cpp
)


target_link_libraries(bench-smoke
    smoke
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)


find_package(PkgConfig REQUIRED)
pkg_check_modules(libxsmm REQUIRED IMPORTED_TARGET libxsmm)

add_executable(bench-libxsmm
    Main.cpp
    GemmLibxsmm.cpp
)

target_link_libraries(bench-libxsmm
    smoke
    benchmark
    openblas
    ${libxsmm_LINK_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
