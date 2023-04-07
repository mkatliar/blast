// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/panel/Potrf.hpp>
#include <blast/math/panel/Gemm.hpp>

#include <blaze/math/dense/DynamicMatrix.h>
#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>

namespace blast :: testing
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

            DynamicPanelMatrix<Real, columnMajor> A(M, M), L(M, M);
            A = blaze_A;

            // Do potrf
            potrf(A, L);

            // Check result
            blaze::DynamicMatrix<Real> const L_blaze = L;
            BLAST_EXPECT_APPROX_EQ(eval(L_blaze * trans(L_blaze)), blaze_A, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    REGISTER_TYPED_TEST_SUITE_P(PanelPotrfTest,
        testDynamicSize
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, PanelPotrfTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, PanelPotrfTest, float);
}