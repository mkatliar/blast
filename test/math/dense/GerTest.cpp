// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blaze/math/TransposeFlag.h>
#define BLAZE_USER_ASSERTION 1

#include <blazefeo/math/dense/Ger.hpp>

#include <blazefeo/Blaze.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    template <typename T>
    class DenseGerTest
    :   public Test
    {
    protected:
        using Real = T;


        void testDynamicImpl()
        {
            for (size_t m = 1; m <= 20; m += 1)
                for (size_t n = 1; n <= 20; n += 1)
                {
                    // Init Blaze matrices
                    //
                    DynamicVector<Real, columnVector> x(m);
                    DynamicVector<Real, rowVector> y(n);
                    DynamicMatrix<Real, columnMajor> A(m, n), B(m, n);
                    randomize(x);
                    randomize(y);
                    randomize(A);

                    Real alpha {};
                    blaze::randomize(alpha);

                    /// Do ger
                    ger(alpha, x, y, A, B);

                    BLAZEFEO_ASSERT_APPROX_EQ(B, evaluate(A + alpha * x * y), 1e-10, 1e-10)
                        << "ger error at size m,n=" << m << "," << n;
                }
        }
    };


    TYPED_TEST_SUITE_P(DenseGerTest);


    TYPED_TEST_P(DenseGerTest, testDynamic)
    {
        this->testDynamicImpl();
    }


    REGISTER_TYPED_TEST_SUITE_P(DenseGerTest
        , testDynamic
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DenseGerTest, double);
}