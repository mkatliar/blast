# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

add_executable(bench-blazefeo
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
    math/dense/Iamax.cpp

    math/panel/StaticGemm.cpp
    math/panel/DynamicGemm.cpp
    math/panel/StaticPotrf.cpp
    math/panel/DynamicPotrf.cpp
)

target_link_libraries(bench-blazefeo
    blazefeo
    bench-blazefeo-common
)
