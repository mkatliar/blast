// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: testing
{
    TEST(SubmatrixTest, testSubmatrixOfStaticPanelMatrix)
    {
        StaticPanelMatrix<double, 12, 12, columnMajor> A;
        auto B = submatrix(A, 4, 0, 8, 8);
        EXPECT_EQ(IsAligned_v<decltype(B)>, false);
    }


    TEST(SubmatrixTest, testSubmatrixOfConstStaticPanelMatrix)
    {
        StaticPanelMatrix<double, 12, 12, columnMajor> const A;
        auto B = submatrix(A, 4, 0, 8, 8);
        EXPECT_EQ(IsAligned_v<decltype(B)>, false);
    }


    TEST(SubmatrixTest, testSubmatrixOfDynamicPanelMatrix)
    {
        DynamicPanelMatrix<double, columnMajor> A(12, 12);
        auto B = submatrix(A, 4, 0, 8, 8);

        static_assert(std::is_same_v<decltype(&B(0, 0)), double *>);
        EXPECT_EQ(IsAligned_v<decltype(B)>, false);
    }


    TEST(SubmatrixTest, testSubmatrixOfConstDynamicPanelMatrix)
    {
        DynamicPanelMatrix<double, columnMajor> const A(12, 12);
        auto B = submatrix(A, 4, 0, 8, 8);

        static_assert(std::is_same_v<decltype(&B(0, 0)), double const *>);
        EXPECT_EQ(IsAligned_v<decltype(B)>, false);
    }


    TEST(SubmatrixTest, testSubmatrixOfSubmatrix)
    {
        DynamicPanelMatrix<double, columnMajor> A(12, 12);
        for (size_t i = 0; i < rows(A); ++i)
            for (size_t j = 0; j < columns(A); ++j)
                A(i, j) = -(100. * i + j);

        auto B = submatrix(A, 4, 0, 8, 8);
        ASSERT_EQ(&B.operand(), &A);

        auto B1 = submatrix(B, 4, 1, 2, 2);
        EXPECT_EQ(&B1.operand(), &A);
        EXPECT_EQ(B1.row(), B.row() + 4);
        EXPECT_EQ(B1.column(), B.column() + 1);

        static_assert(std::is_same_v<decltype(B), decltype(B1)>);

        for (size_t i = 0; i < rows(B1); ++i)
            for (size_t j = 0; j < columns(B1); ++j)
                B1(i, j) = 100. * i + j;

        for (size_t i = 0; i < rows(B1); ++i)
            for (size_t j = 0; j < columns(B1); ++j)
            {
                auto const val = 100. * i + j;
                ASSERT_EQ(B1(i, j), val);
                ASSERT_EQ(A(i + B1.row(), j + B1.column()), val);
            }

        EXPECT_EQ(IsAligned_v<decltype(B)>, false);
    }
}
