// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define BLAST_USER_ASSERTION 1

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/views/submatrix/Panel.hpp>
#include <blast/math/algorithm/Gemm.hpp>
#include <blast/math/algorithm/Randomize.hpp>
#include <blast/math/reference/Gemm.hpp>

#include <test/Testing.hpp>
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
                    DynamicPanelMatrix<Real, columnMajor> A(M, K);
                    DynamicPanelMatrix<Real, columnMajor> B(N, K);
                    DynamicPanelMatrix<Real, columnMajor> C(M, N);
                    DynamicPanelMatrix<Real, columnMajor> D(M, N);

                    randomize(A);
                    randomize(B);
                    randomize(C);

                    // Do gemm with BLAST
                    gemm(A, trans(B), C, D);

                    DynamicPanelMatrix<Real, columnMajor> D_ref(M, N);
                    reference::gemm(1., A, trans(B), 1., C, D_ref);

                    BLAST_ASSERT_APPROX_EQ(D, D_ref, absTol<Real>(), relTol<Real>())
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
