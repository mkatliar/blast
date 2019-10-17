find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-smoke
    Main.cpp
    
    gemm_kernel/Gemm.cpp
    gemm_kernel/Store.cpp
    gemm_kernel/StoreVariableSize.cpp

    Gemm.cpp
    GemmImpl.cpp
)


target_link_libraries(bench-smoke
    smoke
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
