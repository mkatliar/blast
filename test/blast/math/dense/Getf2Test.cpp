// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#include <blast/math/dense/Getf2.hpp>
#include <blast/math/dense/Laswp.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>

#include <vector>


namespace blast :: testing
{
    template <typename T>
    class DenseGetf2Test
    :   public Test
    {
    protected:
        using Real = T;

        template <typename MT, bool SO>
        static blaze::DynamicMatrix<Real> luRestore(blaze::Matrix<MT, SO> const& LU, size_t * ipiv)
        {
            auto const M = rows(LU);
            auto const N = columns(LU);
            auto const K = std::min(M, N);

            blaze::DynamicMatrix<Real> L(M, K, 0.);
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < i && j < K; ++j)
                    L(i, j) = (*LU)(i, j);

                if (i < K)
                    L(i, i) = 1.;
            }

            blaze::DynamicMatrix<Real> U(K, N, 0.);
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
            size_t const S_MAX = 5;

            for (size_t M = 1; M <= S_MAX; ++M)
            {
                for (size_t N = 1; N <= S_MAX; ++N)
                {
                    size_t const K = std::min(M, N);

                    // Init matrices
                    //
                    blaze::DynamicMatrix<Real, SO> A(M, N);
                    randomize(A);
                    blaze::DynamicMatrix<Real, SO> A_orig = A;

                    // Do getrf
                    std::vector<size_t> ipiv(K);
                    blast::getf2(A, ipiv.data());

                    // Check result
                    laswp(A_orig, 0, K, ipiv.data());
                    BLAST_EXPECT_APPROX_EQ(A_orig, luRestore(A, ipiv.data()), absTol<Real>(), relTol<Real>())
                        << "getf2() error for size (" << M << ", " << N << ")";
                }
            }
        }
    };


    TYPED_TEST_SUITE_P(DenseGetf2Test);


    TYPED_TEST_P(DenseGetf2Test, testDynamicRowMajor)
    {
        this->template testDynamic<rowMajor>();
    }


    TYPED_TEST_P(DenseGetf2Test, testDynamicColumnMajor)
    {
        this->template testDynamic<columnMajor>();
    }


    REGISTER_TYPED_TEST_SUITE_P(DenseGetf2Test
        , testDynamicRowMajor
        , testDynamicColumnMajor
        // , testStatic
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DenseGetf2Test, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, DensePotrtTest, float);
}
