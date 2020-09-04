# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(Eigen3 3.3.7 REQUIRED)
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

target_compile_options(bench-eigen
    PRIVATE "-Wno-deprecated-anon-enum-enum-conversion"
)
