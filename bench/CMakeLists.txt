find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-smoke
    Main.cpp
    PanelGemm.cpp
    Gemm.cpp
)


target_link_libraries(bench-smoke
    smoke
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
