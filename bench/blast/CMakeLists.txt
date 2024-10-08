# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

add_executable(bench-blast
    Main.cpp

    math/expressions/AssignDensePanel.cpp
    math/expressions/AssignPanelDense.cpp

    math/simd/Ger.cpp
    math/simd/Load.cpp
    math/simd/Store.cpp
    math/simd/PartialLoad.cpp
    math/simd/PartialStore.cpp
    math/simd/Potrf.cpp
    math/simd/Trsm.cpp
    math/simd/Trmm.cpp
    math/simd/PartialGemm.cpp

    math/dense/DynamicSyrk.cpp
    math/dense/StaticSyrk.cpp
    math/dense/DynamicGemm.cpp
    math/dense/StaticGemm.cpp
    math/dense/StaticPotrf.cpp
    math/dense/StaticGetrf.cpp
    math/dense/StaticTrmm.cpp
    math/dense/StaticIamax.cpp
    math/dense/DynamicIamax.cpp
    math/dense/StaticTrsm.cpp

    math/panel/StaticGemm.cpp
    math/panel/DynamicGemm.cpp
    math/panel/StaticPotrf.cpp
    math/panel/DynamicPotrf.cpp
    math/panel/StaticMatrixPointer.cpp
)

target_compile_definitions(bench-blast
    # Use Blaze without linking to a BLAS library.
    # Blaze is used to prepare data in some of the benchmarks.
    PRIVATE BLAZE_BLAS_MODE=0
)

target_link_libraries(bench-blast
    blast
    bench-blast-common
)
