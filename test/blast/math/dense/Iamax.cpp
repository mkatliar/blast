// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/dense/Iamax.hpp>
#include <blast/blaze/Math.hpp>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: testing
{
    template <typename T>
    class IamaxTest
    :   public Test
    {
    };


    using MyTypes = Types<double, float>;
    TYPED_TEST_SUITE(IamaxTest, MyTypes);


    TYPED_TEST(IamaxTest, testIamax)
    {
        using Real = TypeParam;

        for (size_t n = 1; n < 50; ++n)
        {
            blaze::DynamicVector<Real> x(n);
            randomize(x);

            size_t const ind = iamax(x);

            for (auto v : x)
                ASSERT_LE(std::abs(v), x[ind]) << "Error at size " << n;
        }
    }
}
