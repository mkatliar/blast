// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/dense/Potrf.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>


namespace blazefeo :: testing
{
    template <typename T>
    class DensePotrtTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(DensePotrtTest);


    TYPED_TEST_P(DensePotrtTest, testDynamic)
    {
        using Real = TypeParam;

        for (size_t M = 0; M <= 50; ++M)
        {
            // Init matrices
            //
            DynamicMatrix<Real, columnMajor> A(M, M), L(M, M);
            makePositiveDefinite(A);
            reset(L);

            // Do potrf
            blazefeo::potrf(A, L);

            // Check result
            BLAZEFEO_EXPECT_APPROX_EQ(L * trans(L), A, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    TYPED_TEST_P(DensePotrtTest, testStatic)
    {
        using Real = TypeParam;

        size_t const M = 20;

        // Init matrices
        //
        StaticMatrix<Real, M, M, columnMajor> A, L;
        makePositiveDefinite(A);
        reset(L);

        // Do potrf
        blazefeo::potrf(A, L);

        // Check result
        BLAZEFEO_EXPECT_APPROX_EQ(L * trans(L), A, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
    }


    TYPED_TEST_P(DensePotrtTest, testStaticInplace)
    {
        using Real = TypeParam;

        size_t const M = 3;

        // Init matrices
        //
        StaticMatrix<Real, M, M, columnMajor> A_orig;
        makePositiveDefinite(A_orig);

        StaticMatrix<Real, M, M, columnMajor> A = A_orig;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = i + 1; j < M; ++j)
                reset(A(i, j));

        // Do potrf in place
        blazefeo::potrf(A, A);

        // Check result
        BLAZEFEO_EXPECT_APPROX_EQ(A * trans(A), A_orig, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
    }


    REGISTER_TYPED_TEST_SUITE_P(DensePotrtTest
        , testDynamic
        , testStatic
        , testStaticInplace
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DensePotrtTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, DensePotrtTest, float);
}