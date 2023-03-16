// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blast :: testing
{
    template <typename Real>
    class StaticPanelMatrixTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(StaticPanelMatrixTest);


    TYPED_TEST_P(StaticPanelMatrixTest, testIsPanelMatrix)
    {
        using MatrixType = StaticPanelMatrix<TypeParam, 5, 7>;
        EXPECT_TRUE(IsPanelMatrix_v<MatrixType>);
    }


    TYPED_TEST_P(StaticPanelMatrixTest, testPanels)
    {
        size_t constexpr SS = PanelSize_v<TypeParam>;
        EXPECT_EQ((StaticPanelMatrix<TypeParam, 2 * SS, 1, columnMajor>().panels()), 2);
        EXPECT_EQ((StaticPanelMatrix<TypeParam, 2 * SS + 1, 1, columnMajor>().panels()), 3);
    }


    TYPED_TEST_P(StaticPanelMatrixTest, testElementAccess)
    {
        size_t constexpr SS = PanelSize_v<TypeParam>;
        size_t constexpr M = 2 * SS + 1;
        size_t constexpr N = 3 * SS + 2;

        blaze::StaticMatrix<TypeParam, M, N> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<TypeParam, M, N, columnMajor> A;
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


    TYPED_TEST_P(StaticPanelMatrixTest, testLoad)
    {
        size_t constexpr SS = PanelSize_v<TypeParam>;
        size_t constexpr M = 2 * SS + 1;
        size_t constexpr N = 3 * SS + 2;

        StaticPanelMatrix<TypeParam, M, N, columnMajor> A;
        randomize(A);

        for (size_t i = 0; i < M; i += SS)
            for (size_t j = 0; j < N; ++j)
            {
                auto const xmm = A.template load<SS>(i, j);

                for (size_t k = 0; k < SS; ++k)
                    ASSERT_EQ(xmm[k], A(i + k, j)) << "element mismatch at i,j,k=" << i << "," << j << "," << k;
            }
    }


    TYPED_TEST_P(StaticPanelMatrixTest, testStore)
    {
        size_t constexpr SS = PanelSize_v<TypeParam>;
        size_t constexpr M = 2 * SS + 1;
        size_t constexpr N = 3 * SS + 2;

        StaticPanelMatrix<TypeParam, M, N, columnMajor> A;
        IntrinsicType_t<TypeParam> val;

        for (size_t i = 0; i < SS; ++i)
            val[i] = TypeParam(i + 1);

        for (size_t i = 0; i < M; i += SS)
            for (size_t j = 0; j < N; ++j)
            {
                A = TypeParam(0.);
                A.store(i, j, val);

                for (size_t k = 0; k < SS && i + k < rows(A); ++k)
                    ASSERT_EQ(A(i + k, j), val[k]) << "element mismatch at i,j,k=" << i << "," << j << "," << k;
            }
    }


    TYPED_TEST_P(StaticPanelMatrixTest, testPMatPMatMulAssign)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr K = 10;
        size_t constexpr P = 4;

        StaticPanelMatrix<TypeParam, M, K, columnMajor> A;
        StaticPanelMatrix<TypeParam, K, N, columnMajor> B;
        StaticPanelMatrix<TypeParam, M, N, columnMajor> D;

        randomize(A);
        randomize(B);

        // D = A * B;

        // for (size_t i = 0; i < M; ++i)
        //     for (size_t j = 0; j < N; ++j)
        //         EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(StaticPanelMatrixTest, testPMatTPMatMulAssign)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr K = 10;
        size_t constexpr P = 4;

        StaticPanelMatrix<TypeParam, M, K, columnMajor> A;
        StaticPanelMatrix<TypeParam, N, K, columnMajor> B;
        StaticPanelMatrix<TypeParam, M, N, columnMajor> D;

        randomize(A);
        randomize(B);

        // D = A * trans(B);

        // using E = decltype(A * trans(B));

        // E e(0);

        // using MT2 = decltype(A);
        // using MT3 = decltype(trans(B));
        // static_assert(IsPanelMatrix_v<MT2>);
        // static_assert(IsRowMajorMatrix_v<MT2>);
        // static_assert(IsPanelMatrix_v<MT3>);
        // static_assert(IsRowMajorMatrix_v<MT3>);

        // for (size_t i = 0; i < M; ++i)
        //     for (size_t j = 0; j < N; ++j)
        //         EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    REGISTER_TYPED_TEST_SUITE_P(StaticPanelMatrixTest,
        testIsPanelMatrix,
        testPanels,
        testElementAccess,
        testLoad,
        testStore,
        testPMatPMatMulAssign,
        testPMatTPMatMulAssign
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, StaticPanelMatrixTest, double);
    INSTANTIATE_TYPED_TEST_SUITE_P(float, StaticPanelMatrixTest, float);
}