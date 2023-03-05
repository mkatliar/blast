// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/Trsm.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    TEST(DenseTrsmTest, testLeftLowerNonUnit)
    {
        for (size_t m = 1; m <= 20; ++m)
            for (size_t n = 1; n <= 20; ++n)
            {
                // Init Blaze matrices
                //
                LowerMatrix<DynamicMatrix<double>> A(m, m);
                DynamicMatrix<double> B(m, n);
                DynamicMatrix<double> X(m, n);
                randomize(A);
                A += IdentityMatrix<double>(m); // Improve conditioning
                randomize(B);

                // Do trsv
                trsm<UpLo::Lower, false>(A, B, X);

                BLAZEFEO_ASSERT_APPROX_EQ(evaluate(A * X), B, 1e-10, 1e-10)
                    << "trsm error at size (" << m << ", " << n << ")";
            }
    }


    TEST(DenseTrsmTest, testLeftUpperNonUnit)
    {
        for (size_t m = 1; m <= 20; ++m)
            for (size_t n = 1; n <= 20; ++n)
            {
                // Init Blaze matrices
                //
                UpperMatrix<DynamicMatrix<double>> A(m, m);
                DynamicMatrix<double> B(m, n);
                DynamicMatrix<double> X(m, n);
                randomize(A);
                A += IdentityMatrix<double>(m); // Improve conditioning
                randomize(B);

                // Do trsv
                trsm<UpLo::Upper, false>(A, B, X);

                BLAZEFEO_ASSERT_APPROX_EQ(evaluate(A * X), B, 1e-10, 1e-10)
                    << "trsm error at size (" << m << ", " << n << ")";
            }
    }


    TEST(DenseTrsmTest, testLeftLowerUnit)
    {
        for (size_t m = 1; m <= 20; ++m)
            for (size_t n = 1; n <= 20; ++n)
            {
                // Init Blaze matrices
                //
                LowerMatrix<DynamicMatrix<double>> A(m, m);
                DynamicMatrix<double> B(m, n);
                DynamicMatrix<double> X(m, n);
                randomize(A);
                randomize(B);

                // Do trsv
                trsm<UpLo::Lower, true>(A, B, X);

                diagonal(A) = 1.;
                BLAZEFEO_ASSERT_APPROX_EQ(evaluate(A * X), B, 1e-10, 1e-10)
                    << "trsm error at size (" << m << ", " << n << ")";
            }
    }


    TEST(DenseTrsmTest, testLeftUpperUnit)
    {
        for (size_t m = 1; m <= 20; ++m)
            for (size_t n = 1; n <= 20; ++n)
            {
                // Init Blaze matrices
                //
                UpperMatrix<DynamicMatrix<double>> A(m, m);
                DynamicMatrix<double> B(m, n);
                DynamicMatrix<double> X(m, n);
                randomize(A);
                randomize(B);

                // Do trsv
                trsm<UpLo::Upper, true>(A, B, X);

                diagonal(A) = 1.;
                BLAZEFEO_ASSERT_APPROX_EQ(evaluate(A * X), B, 1e-10, 1e-10)
                    << "trsm error at size (" << m << ", " << n << ")";
            }
    }
}