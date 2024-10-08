// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define BLAST_USER_ASSERTION 1

#include <blast/math/dense/Syrk.hpp>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>

#include <blast/blaze/Math.hpp>


namespace blast :: testing
{
    template <typename T>
    class DenseSyrkTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(DenseSyrkTest);


    TYPED_TEST_P(DenseSyrkTest, testDynamicLn)
    {
        using Real = double;

        for (size_t m = 1; m <= 20; m += 1)
            for (size_t k = 1; k <= 20; ++k)
            {
                // Init Blaze matrices
                //
                blaze::DynamicMatrix<Real, columnMajor> A(m, k);
                blaze::DynamicMatrix<Real, columnMajor> C(m, m), D(m, m);
                randomize(A);
                makeSymmetric(C);
                // for (size_t i = 0; i < m; ++i)
                //     for (size_t j = i + 1; j < m; ++j)
                //         C(i, j) = 0.;

                // std::cout << "A=\n" << A << std::endl;
                // std::cout << "B=\n" << B << std::endl;
                // std::cout << "C=\n" << C << std::endl;
                // std::cout << "C+A*trans(B)=\n" << C + A * trans(B) << std::endl;

                // D = alpha * A * A^T + beta * C; C, D lower triangular
                Real const alpha = 1.0;
                Real const beta = 1.0;
                D = 0.;
                syrkLower(alpha, A, beta, C, D);

                // Calculate the reference value
                auto D_ref = evaluate(alpha * A * trans(A) + beta * C);
                for (size_t i = 0; i < m; ++i)
                    for (size_t j = i + 1; j < m; ++j)
                        D_ref(i, j) = 0.;

                // Print the result
                // std::cout << "D=\n" << D;

                BLAST_ASSERT_APPROX_EQ(D, D_ref, 1e-10, 1e-10)
                    << "syrk error at size m,k=" << m << "," << k;
            }
    }

    REGISTER_TYPED_TEST_SUITE_P(DenseSyrkTest
        ,   testDynamicLn
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DenseSyrkTest, double);
}
