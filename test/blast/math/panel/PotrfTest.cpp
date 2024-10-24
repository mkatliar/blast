// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/dense/DynamicMatrix.hpp>
#include <blast/math/panel/Potrf.hpp>
#include <blast/math/algorithm/Gemm.hpp>
#include <blast/math/algorithm/Randomize.hpp>
#include <blast/math/reference/Gemm.hpp>

#include <test/Testing.hpp>
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
            DynamicPanelMatrix<Real, columnMajor> A(M, M), L(M, M), A1(M, M);
            makePositiveDefinite(A);

            // Do potrf
            potrf(A, L);

            // Check that L is lower-triangular
            for (std::size_t i = 0; i < rows(L); ++i)
                for (std::size_t j = i + 1; j < columns(L); ++j)
                    EXPECT_NEAR(L(i, j), Real {}, absTol<Real>());

            // Check A == L * trans(L)
            reset(A1);
            reference::gemm(1., L, trans(L), 0., A1, A1);

            BLAST_EXPECT_APPROX_EQ(A1, A, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    REGISTER_TYPED_TEST_SUITE_P(PanelPotrfTest,
        testDynamicSize
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, PanelPotrfTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, PanelPotrfTest, float);
}
