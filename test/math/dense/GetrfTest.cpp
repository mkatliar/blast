// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "blazefeo/math/dense/Laswp.hpp"
#include <blaze/math/StorageOrder.h>
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
    protected:
        using Real = T;

        template <typename MT, bool SO>
        static DynamicMatrix<Real> luRestore(Matrix<MT, SO> const& LU, size_t * ipiv)
        {
            auto const M = rows(LU);
            auto const N = columns(LU);
            auto const K = std::min(M, N);

            DynamicMatrix<Real> L(M, K, 0.);
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < i && j < K; ++j)
                    L(i, j) = (*LU)(i, j);

                if (i < K)
                    L(i, i) = 1.;
            }

            DynamicMatrix<Real> U(K, N, 0.);
            for (size_t i = 0; i < K; ++i)
            {
                for (size_t j = i; j < N; ++j)
                    U(i, j) = (*LU)(i, j);
            }

            return L * U;
        }


        template <bool SO>
        void testDynamic()
        {
            for (size_t M = 0; M <= 20; ++M)
            {
                for (size_t N = 0; N <= 20; ++N)
                {
                    size_t const K = std::min(M, N);

                    // Init matrices
                    //
                    DynamicMatrix<Real, SO> A(M, N);
                    randomize(A);
                    DynamicMatrix<Real, SO> A_orig = A;

                    // Do getrf
                    std::vector<size_t> ipiv(K);
                    blazefeo::getrf(A, ipiv.data());

                    // Check result
                    laswp(A_orig, 0, K, ipiv.data());
                    BLAZEFEO_EXPECT_APPROX_EQ(A_orig, luRestore(A, ipiv.data()), absTol<Real>(), relTol<Real>())
                        << "getrf() error for size (" << M << ", " << N << ")";
                }
            }
        }
    };


    TYPED_TEST_SUITE_P(DenseGetrfTest);


    TYPED_TEST_P(DenseGetrfTest, testDynamicRowMajor)
    {
        this->template testDynamic<rowMajor>();
    }


    TYPED_TEST_P(DenseGetrfTest, testDynamicColumnMajor)
    {
        this->template testDynamic<columnMajor>();
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
        , testDynamicRowMajor
        , testDynamicColumnMajor
        // , testStatic
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DenseGetrfTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, DensePotrtTest, float);
}