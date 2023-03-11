// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define BLAZE_USER_ASSERTION 1

#include <blazefeo/math/dense/Gemm.hpp>

#include <blazefeo/Blaze.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    template <typename T>
    class DenseGemmTest
    :   public Test
    {
    protected:
        using Real = T;


        template <bool SOA, bool SOB>
        void testAlignedImpl()
        {
            for (size_t m = 1; m <= 20; m += 1)
                for (size_t n = 1; n <= 20; n += 1)
                    for (size_t k = 1; k <= 20; ++k)
                    {
                        // Init Blaze matrices
                        //
                        DynamicMatrix<Real, SOA> A(m, k), C(m, n), D(m, n);
                        DynamicMatrix<Real, SOB> B(k, n);
                        randomize(A);
                        randomize(B);
                        randomize(C);

                        Real alpha {}, beta {};
                        blaze::randomize(alpha);
                        blaze::randomize(beta);

                        /// Do gemm
                        gemm(alpha, A, B, beta, C, D);

                        BLAZEFEO_ASSERT_APPROX_EQ(D, evaluate(beta * C + alpha * A * B), 1e-10, 1e-10)
                            << "gemm error at size m,n,k=" << m << "," << n << "," << k;
                    }
        }


        template <bool SOA, bool SOB>
        void testUnalignedImpl()
        {
            size_t constexpr S_MAX = 20;
            DynamicMatrix<Real, SOA> AA(S_MAX, S_MAX), CC(S_MAX, S_MAX), DD(S_MAX, S_MAX);
            DynamicMatrix<Real, SOB> BB(S_MAX, S_MAX);

            for (size_t m = 1; m <= S_MAX; m += 1)
                for (size_t n = 1; n <= S_MAX; n += 1)
                    for (size_t k = 1; k <= S_MAX; ++k)
                    {
                        // Init Blaze matrices
                        //
                        auto A = submatrix(AA, rows(AA) - m, columns(AA) - k, m, k);
                        auto C = submatrix(CC, rows(CC) - m, columns(CC) - n, m, n);
                        auto D = submatrix(DD, rows(DD) - m, columns(DD) - n, m, n);
                        auto B = submatrix(BB, rows(BB) - k, columns(BB) - n, k, n);
                        randomize(A);
                        randomize(B);
                        randomize(C);

                        Real alpha {}, beta {};
                        blaze::randomize(alpha);
                        blaze::randomize(beta);

                        /// Do gemm
                        gemm(alpha, A, B, beta, C, D);

                        BLAZEFEO_ASSERT_APPROX_EQ(D, evaluate(beta * C + alpha * A * B), 1e-10, 1e-10)
                            << "gemm error at size m,n,k=" << m << "," << n << "," << k;
                    }
        }
    };


    TYPED_TEST_SUITE_P(DenseGemmTest);


    TYPED_TEST_P(DenseGemmTest, testAlignedCr)
    {
        this->template testAlignedImpl<columnMajor, rowMajor>();
    }


    TYPED_TEST_P(DenseGemmTest, testAlignedCc)
    {
        this->template testAlignedImpl<columnMajor, columnMajor>();
    }


    TYPED_TEST_P(DenseGemmTest, testUnalignedCr)
    {
        this->template testUnalignedImpl<columnMajor, rowMajor>();
    }


    TYPED_TEST_P(DenseGemmTest, testUnalignedCc)
    {
        this->template testUnalignedImpl<columnMajor, columnMajor>();
    }


    REGISTER_TYPED_TEST_SUITE_P(DenseGemmTest
        , testAlignedCr
        , testAlignedCc
        , testUnalignedCr
        , testUnalignedCc
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DenseGemmTest, double);
}