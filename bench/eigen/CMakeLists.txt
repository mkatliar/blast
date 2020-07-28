find_package(Eigen3 3.2.10 REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-eigen
    Main.cpp
    Potrf.cpp
    Syrk.cpp
    Gemm.cpp
)


target_include_directories(bench-eigen
    PUBLIC ${EIGEN3_INCLUDE_DIR}
)


target_link_libraries(bench-eigen
    bench-blazefeo-common
    ${CMAKE_THREAD_LIBS_INIT}
)
