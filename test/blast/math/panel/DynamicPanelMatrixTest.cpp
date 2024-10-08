// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>

#include <blaze/Math.h>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: testing
{
    TEST(DynamicPanelMatrixTest, testIsStatic)
    {
        using MatrixType = DynamicPanelMatrix<double>;
        ASSERT_FALSE(IsStatic_v<MatrixType>);
    }


    TEST(DynamicPanelMatrixTest, testIsAligned)
    {
        using MatrixType = DynamicPanelMatrix<double>;
        ASSERT_TRUE(IsAligned_v<MatrixType>);
    }


    TEST(DynamicPanelMatrixTest, testIsPadded)
    {
        using MatrixType = DynamicPanelMatrix<double>;
        ASSERT_TRUE(IsPadded_v<MatrixType>);
    }


    TEST(DynamicPanelMatrixTest, testSpacing)
    {
        {
            DynamicPanelMatrix<double> m(5, 2);
            EXPECT_EQ(m.spacing(), 4 * 2);
        }

        {
            DynamicPanelMatrix<double> m(5, 4);
            EXPECT_EQ(m.spacing(), 4 * 4);
        }

        {
            DynamicPanelMatrix<double> m(5, 7);
            EXPECT_EQ(m.spacing(), 4 * 7);
        }
    }


    TEST(DynamicPanelMatrixTest, testElementAccess)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;

        blaze::DynamicMatrix<double> A_ref(M, N);
        randomize(A_ref);

        DynamicPanelMatrix<double> A(M, N);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                A(i, j) = A_ref(i, j);

        auto const& A_cref = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
            {
                EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
                EXPECT_EQ(A_cref(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
            }
    }


    TEST(DynamicPanelMatrixTest, testPack)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;

        blaze::DynamicMatrix<double, blaze::columnMajor> A_ref(M, N);
        randomize(A_ref);

        DynamicPanelMatrix<double> A(M, N);
        A = A_ref;

        auto const& A_cref = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TEST(DynamicPanelMatrixTest, testUnpack)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;

        DynamicPanelMatrix<double> A(M, N);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                blaze::randomize(A(i, j));

        blaze::DynamicMatrix<double, blaze::columnMajor> A1(M, N);
        A1 = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }
}
