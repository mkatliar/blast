find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-blazefeo
    Main.cpp
    
    gemm_kernel/Gemm.cpp
    gemm_kernel/Store.cpp
    gemm_kernel/StoreVariableSize.cpp

    GemmStatic.cpp
    GemmDynamic.cpp
)


target_link_libraries(bench-blazefeo
    blazefeo
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)


find_package(PkgConfig REQUIRED)
pkg_check_modules(libxsmm REQUIRED IMPORTED_TARGET libxsmm)
pkg_check_modules(openblas REQUIRED IMPORTED_TARGET openblas)

add_executable(bench-libxsmm
    Main.cpp
    GemmLibxsmm.cpp
)

target_link_libraries(bench-libxsmm
    blazefeo
    benchmark
    PkgConfig::openblas
    PkgConfig::libxsmm
    ${CMAKE_THREAD_LIBS_INIT}
)
