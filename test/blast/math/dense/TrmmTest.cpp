// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/algorithm/Trmm.hpp>
#include <blast/math/dense/DynamicMatrix.hpp>
#include <blast/math/reference/Trmm.hpp>
#include <blast/math/algorithm/Randomize.hpp>

#include <test/Testing.hpp>


namespace blast :: testing
{
    TEST(DenseTrmmTest, testLeftUpper)
    {
        for (size_t m = 1; m <= 20; ++m)
            for (size_t n = 1; n <= 20; ++n)
            {
                DynamicMatrix<double, columnMajor> A(m, m);
                DynamicMatrix<double, rowMajor> B(m, n);
                DynamicMatrix<double, columnMajor> C(m, n);
                randomize(A);
                randomize(B);

                double alpha {};
                randomize(alpha);

                // Do trmm
                trmm(alpha, A, UpLo::Upper, false, B, C);

                DynamicMatrix<double, columnMajor> C_ref(m, n);
                reference::trmm(alpha, A, UpLo::Upper, false, B, C_ref);
                BLAST_ASSERT_APPROX_EQ(C, C_ref, 1e-10, 1e-10)
                    << "trmm error at size m,n=" << m << "," << n;
            }
    }


    TEST(DenseTrmmTest, testRightLower)
    {
        for (size_t m = 1; m <= 20; ++m)
            for (size_t n = 1; n <= 20; ++n)
            {
                DynamicMatrix<double, columnMajor> A(n, n);
                DynamicMatrix<double, columnMajor> B(m, n);
                DynamicMatrix<double, columnMajor> C(m, n);
                randomize(A);
                randomize(B);

                double alpha {};
                randomize(alpha);

                // Do trmm
                trmm(alpha, B, A, UpLo::Lower, false, C);

                DynamicMatrix<double, columnMajor> C_ref(m, n);
                reference::trmm(alpha, B, A, UpLo::Lower, false, C_ref);
                BLAST_ASSERT_APPROX_EQ(C, C_ref, 1e-10, 1e-10)
                    << "trmm error at size m,n=" << m << "," << n;
            }
    }
}
