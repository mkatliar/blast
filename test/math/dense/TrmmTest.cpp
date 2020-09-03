// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/Trmm.hpp>
#include <blazefeo/math/dense/Gemm.hpp>

#include <blazefeo/Blaze.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(DenseTrmmTest, testLeftUpper)
    {
        for (size_t m = 1; m <= 20; m += 1)
            for (size_t n = 1; n <= 20; n += 1)
            {
                // Init Blaze matrices
                //
                DynamicMatrix<double, columnMajor> A(m, m);
                DynamicMatrix<double, rowMajor> B(m, n);
                DynamicMatrix<double, columnMajor> C(m, n);
                randomize(A);
                randomize(B);

                // Reset lower-triangular part of A
                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < i; ++j)
                        reset(A(i, j));

                double alpha {};
                blaze::randomize(alpha);

                // Do trmm
                trmmLeftUpper(alpha, A, B, C);

                BLAZEFEO_ASSERT_APPROX_EQ(C, evaluate(alpha * A * B), 1e-10, 1e-10)
                    << "trmm error at size m,n=" << m << "," << n;
            }
    }


    TEST(DenseTrmmTest, testRightLower)
    {
        for (size_t m = 4; m <= 20; m += 1)
            for (size_t n = 4; n <= 20; n += 1)
            {
                // Init Blaze matrices
                //
                DynamicMatrix<double, columnMajor> A(n, n);
                DynamicMatrix<double, columnMajor> B(m, n);
                DynamicMatrix<double, columnMajor> C(m, n);
                randomize(A);
                randomize(B);

                // Reset upper-triangular part of A
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = i + 1; j < n; ++j)
                        reset(A(i, j));

                double alpha {};
                blaze::randomize(alpha);

                // Do trmm
                trmmRightLower(alpha, B, A, C);

                BLAZEFEO_ASSERT_APPROX_EQ(C, evaluate(alpha * B * A), 1e-10, 1e-10)
                    << "trmm error at size m,n=" << m << "," << n;
            }
    }
}