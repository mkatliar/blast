# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#
# test
#
find_package(GTest 1.9.0 REQUIRED)
find_package(blaze 3.5 REQUIRED)
find_package(LAPACK REQUIRED)

# ~~~~~~~~~~ Our tests ~~~~~~~~~~~~~~
add_executable(test-blast
    math/simd/RegisterMatrixTest.cpp
    math/simd/DynamicRegisterMatrixTest.cpp
    math/simd/HsumTest.cpp
    math/simd/SimdVecTest.cpp

    math/dense/StaticMatrixPointerTest.cpp
    math/dense/StaticVectorPointerTest.cpp
    math/dense/DynamicMatrixPointerTest.cpp
    math/dense/DynamicVectorPointerTest.cpp
    # math/dense/MatrixPointerTest.cpp
    math/dense/GerTest.cpp
    math/dense/GemmTest.cpp
    math/dense/SyrkTest.cpp
    math/dense/PotrfTest.cpp
    math/dense/GetrfTest.cpp
    math/dense/Getf2Test.cpp
    math/dense/TrmmTest.cpp
    math/dense/TrsvTest.cpp
    math/dense/TrsmTest.cpp
    math/dense/Iamax.cpp

    math/panel/StaticPanelMatrixTest.cpp
    math/panel/StaticPanelMatrixPointerTest.cpp
    math/panel/DynamicPanelMatrixTest.cpp
    math/panel/DynamicPanelMatrixPointerTest.cpp
    math/panel/GemmTest.cpp
    math/panel/PotrfTest.cpp

    math/expressions/PMatTransExprTest.cpp
    math/expressions/AssignPanelDenseTest.cpp
    math/expressions/AssignDensePanelTest.cpp

    math/views/RowTest.cpp
    math/views/SubmatrixTest.cpp
)

target_link_libraries(test-blast PRIVATE
    blast
    blaze::blaze
    GTest::GTest
    GTest::Main
    LAPACK::LAPACK
)

target_compile_definitions(test-blast
    PRIVATE "BLAZE_USER_ASSERTION=1;BLAZE_INTERNAL_ASSERTION=1"
)

gtest_discover_tests(test-blast)
