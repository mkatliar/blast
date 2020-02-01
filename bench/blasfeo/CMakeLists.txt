find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


add_executable(bench-blasfeo
    Main.cpp
    Trmm.cpp
    Gemm.cpp
    SyrkPotrf.cpp
    Potrf.cpp
)


target_link_libraries(bench-blasfeo
    blazefeo-blasfeo
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
