// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/dense/MatrixPointer.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>


namespace blazefeo :: testing
{
    template <typename Ker>
    class DynamicRegisterMatrixTest
    :   public Test
    {
    };


    using MyTypes = Types<
        DynamicRegisterMatrix<double, 4, 4, columnMajor>,
        DynamicRegisterMatrix<double, 4, 2, columnMajor>,
        DynamicRegisterMatrix<double, 4, 1, columnMajor>,
        DynamicRegisterMatrix<double, 8, 4, columnMajor>,
        DynamicRegisterMatrix<double, 12, 4, columnMajor>,
        DynamicRegisterMatrix<float, 8, 4, columnMajor>,
        DynamicRegisterMatrix<float, 16, 4, columnMajor>,
        DynamicRegisterMatrix<float, 24, 4, columnMajor>
    >;


    TYPED_TEST_SUITE(DynamicRegisterMatrixTest, MyTypes);


    TYPED_TEST(DynamicRegisterMatrixTest, testDefaultCtor)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        for (size_t m = 1; m < RM::maxRows(); ++m)
            for (size_t n = 1; n < RM::maxColumns(); ++n)
            {
                RM ker(m, n);
                ASSERT_EQ(ker.rows(), m);
                ASSERT_EQ(ker.columns(), n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        ASSERT_EQ(ker(i, j), ET(0.));
            }
    }


    TYPED_TEST(DynamicRegisterMatrixTest, testReset)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;


        for (size_t m = 1; m < RM::maxRows(); ++m)
            for (size_t n = 1; n < RM::maxColumns(); ++n)
            {
                DynamicMatrix<ET, RM::storageOrder> A(m, n);
                randomize(A);

                RM ker(m, n);
                ker.load(ptr<aligned>(A, 0, 0));

                for (size_t i = 0; i < ker.rows(); ++i)
                    for (size_t j = 0; j < ker.columns(); ++j)
                        ASSERT_EQ(ker(i, j), A(i, j));

                ker.reset();
                for (size_t i = 0; i < ker.rows(); ++i)
                    for (size_t j = 0; j < ker.columns(); ++j)
                        ASSERT_EQ(ker(i, j), ET(0.));
            }
    }


    TYPED_TEST(DynamicRegisterMatrixTest, testLoad)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        for (size_t m = 1; m < RM::maxRows(); ++m)
            for (size_t n = 1; n < RM::maxColumns(); ++n)
            {
                DynamicMatrix<ET, RM::storageOrder> A(m, n);
                randomize(A);

                RM ker(m, n);
                ker.load(ptr<aligned>(A, 0, 0));

                EXPECT_EQ(ker, A);
            }
    }


    TYPED_TEST(DynamicRegisterMatrixTest, testLoadStore)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        for (size_t m = 1; m < RM::maxRows(); ++m)
            for (size_t n = 1; n < RM::maxColumns(); ++n)
            {
                DynamicMatrix<ET, RM::storageOrder> A(m, n), B(m, n);
                randomize(A);

                RM ker(m, n);
                ker.load(ptr<aligned>(A, 0, 0));
                ker.store(ptr<aligned>(B, 0, 0));

                EXPECT_EQ(B, A);
            }
    }


    TYPED_TEST(DynamicRegisterMatrixTest, testGerNn)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        for (size_t m = 1; m < RM::maxRows(); ++m)
            for (size_t n = 1; n < RM::maxColumns(); ++n)
            {
                DynamicMatrix<ET, RM::storageOrder> A(m, 1);
                DynamicMatrix<ET, RM::storageOrder> B(1, n);
                DynamicMatrix<ET, RM::storageOrder> C(m, n);

                randomize(A);
                randomize(B);
                randomize(C);

                ET alpha {};
                blaze::randomize(alpha);

                TypeParam ker(m, n);
                ker.load(ptr<aligned>(C, 0, 0));
                ker.ger(alpha, ptr<aligned>(A, 0, 0), ptr<aligned>(B, 0, 0));

                BLAZEFEO_EXPECT_APPROX_EQ(ker,
                    evaluate(C + alpha * A * B), absTol<ET>(), relTol<ET>());
            }
    }


    TYPED_TEST(DynamicRegisterMatrixTest, testGerNt)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        for (size_t m = 1; m < RM::maxRows(); ++m)
            for (size_t n = 1; n < RM::maxColumns(); ++n)
            {
                DynamicMatrix<ET, RM::storageOrder> A(m, 1);
                DynamicMatrix<ET, !RM::storageOrder> B(1, n);
                DynamicMatrix<ET, RM::storageOrder> C(m, n);

                randomize(A);
                randomize(B);
                randomize(C);

                ET alpha {};
                blaze::randomize(alpha);

                TypeParam ker(m, n);
                ker.load(ptr(C));
                ker.ger(alpha, ~ptr(A), ~ptr(B));

                BLAZEFEO_EXPECT_APPROX_EQ(ker,
                    evaluate(C + alpha * A * B), absTol<ET>(), relTol<ET>());
            }
    }


    // TYPED_TEST(DynamicRegisterMatrixTest, testPotrf)
    // {
    //     using Traits = DynamicRegisterMatrixTraits<TypeParam>;
    //     using ET = typename Traits::ElementType;
    //     static size_t constexpr m = Traits::rows;
    //     static size_t constexpr n = Traits::columns;

    //     if constexpr (m >= n)
    //     {
    //         StaticPanelMatrix<ET, m, n, columnMajor> A, L;
    //         StaticPanelMatrix<ET, m, m, columnMajor> A1;

    //         {
    //             StaticMatrix<ET, n, n, columnMajor> C0;
    //             makePositiveDefinite(C0);

    //             StaticMatrix<ET, m, n, columnMajor> C;
    //             submatrix(C, 0, 0, n, n) = C0;
    //             randomize(submatrix(C, n, 0, m - n, n));

    //             A = C;
    //         }

    //         {
    //             TypeParam ker;
    //             load(ker, A.ptr<aligned>(0, 0), A.spacing());
    //             ker.potrf();
    //             store(ker, L.ptr<aligned>(0, 0), L.spacing());
    //         }

    //         A1 = 0.;
    //         gemm_nt(L, L, A1, A1);

    //         // std::cout << "A=\n" << A << std::endl;
    //         // std::cout << "L=\n" << L << std::endl;
    //         // std::cout << "A1=\n" << A1 << std::endl;

    //         BLAZEFEO_ASSERT_APPROX_EQ(submatrix(A1, 0, 0, m, n), A, absTol<ET>(), relTol<ET>());
    //     }
    //     else
    //     {
    //         std::clog << "DynamicRegisterMatrixTest.testPotrf not implemented for kernels with columns more than rows!" << std::endl;
    //     }
    // }
}