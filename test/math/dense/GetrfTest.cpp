// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blaze/math/adaptors/lowermatrix/BaseTemplate.h>
#include <blaze/math/dense/DynamicMatrix.h>
#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/dense/Getrf.hpp>

#include <cstddef>
#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>

#include <vector>


namespace blazefeo :: testing
{
    template <typename T>
    class DenseGetrfTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(DenseGetrfTest);


    TYPED_TEST_P(DenseGetrfTest, testDynamic)
    {
        using Real = TypeParam;
        using DMat = DynamicMatrix<Real, rowMajor>;

        for (size_t M = 0; M <= 5; ++M)
        {
            for (size_t N = 0; N <= 5; ++N)
            {
                size_t const K = std::min(M, N);

                // Init matrices
                //
                DMat A(M, N);
                randomize(A);
                std::cout << "A=\n" << A;
                DMat const A_orig = A;

                // Do getrf
                std::vector<int> ipiv(K);
                blazefeo::getrf(A, ipiv.data());

                // Check result
                DMat L(M, K, 0.);
                for (size_t i = 0; i < M; ++i)
                {
                    for (size_t j = 0; j < i && j < K; ++j)
                        L(i, j) = A(i, j);

                    if (i < K)
                        L(i, i) = 1.;
                }

                DMat U(K, N, 0.);
                for (size_t i = 0; i < K; ++i)
                {
                    for (size_t j = i; j < N; ++j)
                        U(i, j) = A(i, j);
                }

                // std::cout << "L=\n" << L;
                // std::cout << "U=\n" << U;
                // std::cout << "A-L*U=\n" << A_orig - L * U;

                BLAZEFEO_EXPECT_APPROX_EQ(A_orig, L * U, absTol<Real>(), relTol<Real>()) << "getrf error for size " << M;
            }
        }
    }


    // TYPED_TEST_P(DenseGetrfTest, testStatic)
    // {
    //     using Real = TypeParam;

    //     size_t const M = 20;

    //     // Init matrices
    //     //
    //     StaticMatrix<Real, M, M, columnMajor> A, L;
    //     makePositiveDefinite(A);
    //     reset(L);

    //     // Do potrf
    //     blazefeo::potrf(A, L);
    //     // std::cout << "L=\n" << L << std::endl;

    //     // Check result
    //     DynamicMatrix<Real> L1;
    //     llh(A, L1);
    //     BLAZEFEO_EXPECT_APPROX_EQ(L, L1, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
    // }


    REGISTER_TYPED_TEST_SUITE_P(DenseGetrfTest
        , testDynamic
        // , testStatic
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DenseGetrfTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, DensePotrtTest, float);
}