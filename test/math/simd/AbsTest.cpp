// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blaze/math/AlignmentFlag.h>
#include <blazefeo/math/simd/Simd.hpp>

#include <test/Testing.hpp>

#include <array>
#include <cmath>


using namespace blaze;


namespace blazefeo :: testing
{
    template <typename T>
    class AbsTest
    :   public Test
    {
    };


    using MyTypes = Types<double, float>;
    TYPED_TEST_SUITE(AbsTest, MyTypes);


    TYPED_TEST(AbsTest, testAbs)
    {
        using Scalar = TypeParam;
        size_t constexpr SS = Simd<Scalar>::size;

        std::array<Scalar, SS> a;
        for (size_t i = 0; i < SS; ++i)
            a[i] = Scalar(0.1) * i * (i % 2 ? -1 : 1);

        auto const v = load<unaligned>(a.data());
        auto const abs_v = abs(v);

        for (size_t i = 0; i < SS; ++i)
            ASSERT_EQ(abs_v[i], std::abs(a[i]));
    }
}