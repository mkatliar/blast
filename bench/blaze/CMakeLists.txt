find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-blaze
    Main.cpp
    ZeroMatrixAssign.cpp
    Syrk.cpp
    Trmm.cpp
    Trmv.cpp
    Trsv.cpp
    Gemm.cpp
    GemmSubmatrix.cpp
    SqrNorm.cpp
    LowerMatrixScalarMultiply.cpp
    Column.cpp
)


target_link_libraries(bench-blaze
    blaze::blaze
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
