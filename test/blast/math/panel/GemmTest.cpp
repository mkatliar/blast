// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define BLAST_USER_ASSERTION 1

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/views/submatrix/Panel.hpp>
#include <blast/math/algorithm/Gemm.hpp>

#include <blaze/Math.h>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>
#include <test/Tolerance.hpp>


namespace blast :: testing
{
    template <typename T>
    class PanelGemmTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(PanelGemmTest);


    TYPED_TEST_P(PanelGemmTest, testNT)
    {
        using Real = TypeParam;
        size_t const M_max = 20, N_max = 20, K_max = 20;

        for (size_t M = 1; M <= M_max; ++M)
        {
            for (size_t N = 1; N <= N_max; ++N)
            {
                for (size_t K = 1; K <= K_max; ++K)
                {
                    // Init Blaze matrices
                    //
                    blaze::DynamicMatrix<Real, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
                    randomize(blaze_A);
                    randomize(blaze_B);
                    randomize(blaze_C);

                    // Init BLAST matrices
                    //
                    DynamicPanelMatrix<Real> A(M, K);
                    DynamicPanelMatrix<Real> B(N, K);
                    DynamicPanelMatrix<Real> C(M, N);
                    DynamicPanelMatrix<Real> D(M, N);

                    A = blaze_A;
                    B = blaze_B;
                    C = blaze_C;

                    // Do gemm with BLAST
                    gemm(A, trans(B), C, D);

                    // Copy the resulting D matrix from BLAST to Blaze
                    blaze_D = D;

                    BLAST_ASSERT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), absTol<Real>(), relTol<Real>())
                        << "gemm error at size m,n,k=" << M << "," << N << "," << K;
                }
            }
        }
    }



    REGISTER_TYPED_TEST_SUITE_P(PanelGemmTest,
        testNT
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, PanelGemmTest, double);
    INSTANTIATE_TYPED_TEST_SUITE_P(float, PanelGemmTest, float);
}
