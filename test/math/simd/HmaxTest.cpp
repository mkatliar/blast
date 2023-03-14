// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/simd/Simd.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

#include <array>
#include <algorithm>


using namespace blaze;


namespace blazefeo :: testing
{
    template <typename T>
    class HmaxTest
    :   public Test
    {
    };


    using MyTypes = Types<double, float>;
    TYPED_TEST_SUITE(HmaxTest, MyTypes);


    TYPED_TEST(HmaxTest, testHmax)
    {
        using Scalar = TypeParam;
        size_t constexpr SS = Simd<Scalar>::size;

        std::array<Scalar, SS> a;

        for (int i = 0; i < 1000; ++i)
        {
            randomize(a);
            Scalar const max_v = hmax(load<unaligned>(a.data()));
            ASSERT_EQ(max_v, *std::max_element(a.begin(), a.end()));
        }
    }
}