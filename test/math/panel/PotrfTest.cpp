// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>

namespace blazefeo :: testing
{
    template <typename T>
    class PanelPotrfTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(PanelPotrfTest);


    TYPED_TEST_P(PanelPotrfTest, testDynamicSize)
    {
        using Real = TypeParam;

        for (size_t M = 0; M <= 50; ++M)
        {
            // Init matrices
            //
            DynamicMatrix<Real, columnMajor> blaze_A(M, M);
            makePositiveDefinite(blaze_A);

            DynamicPanelMatrix<Real, columnMajor> A(M, M), L(M, M), A1(M, M);
            A = blaze_A;

            // Do potrf
            potrf(A, L);

            // Check result
            A1 = 0.;
            gemm_nt(L, L, A1, A1);

            BLAZEFEO_EXPECT_APPROX_EQ(A1, A, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    REGISTER_TYPED_TEST_SUITE_P(PanelPotrfTest,
        testDynamicSize
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, PanelPotrfTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, PanelPotrfTest, float);
}