find_package(Threads REQUIRED)


add_executable(bench-blasfeo
    Main.cpp
    Trmm.cpp
    Gemm.cpp
    Syrk.cpp
    SyrkPotrf.cpp
    Potrf.cpp
)


target_link_libraries(bench-blasfeo
    blazefeo-blasfeo
    bench-blazefeo-common
    ${CMAKE_THREAD_LIBS_INIT}
)
