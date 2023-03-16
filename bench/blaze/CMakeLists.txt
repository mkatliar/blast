# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-blaze
    Main.cpp
    ZeroMatrixAssign.cpp
    Syrk.cpp
    DynamicTrmm.cpp
    StaticTrmm.cpp
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
    bench-blast-common
    ${CMAKE_THREAD_LIBS_INIT}
)
