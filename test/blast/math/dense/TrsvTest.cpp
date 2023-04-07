// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/dense/Trsv.hpp>

#include <test/Testing.hpp>

#include <blaze/Math.h>


namespace blast :: testing
{
    TEST(DenseTrsvTest, testLeftLower)
    {
        for (size_t n = 1; n <= 20; ++n)
        {
            // Init Blaze matrices
            //
            blaze::LowerMatrix<blaze::DynamicMatrix<double>> A(n, n);
            blaze::DynamicVector<double> b(n);
            blaze::DynamicVector<double> x(n);
            randomize(A);
            A += blaze::IdentityMatrix<double>(n); // Improve conditioning
            randomize(b);

            // Do trsv
            trsvLeftLower(A, b, x);

            BLAST_ASSERT_APPROX_EQ(evaluate(A * x), b, 1e-10, 1e-10)
                << "trsv error at size n=" << n;
        }
    }


    TEST(DenseTrsvTest, testLeftUpper)
    {
        for (size_t n = 1; n <= 20; ++n)
        {
            // Init Blaze matrices
            //
            blaze::UpperMatrix<blaze::DynamicMatrix<double>> A(n, n);
            blaze::DynamicVector<double> b(n);
            blaze::DynamicVector<double> x(n);
            randomize(A);
            A += blaze::IdentityMatrix<double>(n); // Improve conditioning
            randomize(b);

            // Do trsv
            trsvLeftUpper(A, b, x);

            BLAST_ASSERT_APPROX_EQ(evaluate(A * x), b, 1e-10, 1e-10)
                << "trsv error at size n=" << n;
        }
    }
}