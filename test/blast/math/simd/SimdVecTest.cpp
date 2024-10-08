// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/Simd.hpp>

#include <blaze/Math.h>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>

#include <array>
#include <algorithm>


using namespace blaze;


namespace blast :: testing
{
    template <typename T>
    class SimdVecTest
    :   public Test
    {
    };


    using MyTypes = Types<double, float>;
    TYPED_TEST_SUITE(SimdVecTest, MyTypes);


    TYPED_TEST(SimdVecTest, testHmax)
    {
        using Scalar = TypeParam;
        size_t constexpr SS = SimdSize_v<Scalar>;

        std::array<Scalar, SS> a;
        for (int i = 0; i < 1000; ++i)
        {
            randomize(a);
            SimdVec<Scalar> v {data(a), false};
            ASSERT_EQ(max(v), *std::max_element(a.begin(), a.end()));
        }
    }


    TYPED_TEST(SimdVecTest, testAbs)
    {
        using Scalar = TypeParam;
        size_t constexpr SS = SimdSize_v<Scalar>;

        blaze::StaticVector<Scalar, SS> a;
        randomize(a);
        a -= 0.5;

        SimdVec<Scalar> const v {a.data(), false};
        SimdVec<Scalar> abs_v = abs(v);

        for (size_t i = 0; i < SS; ++i)
            ASSERT_EQ(abs_v[i], std::abs(a[i]));
    }


    TYPED_TEST(SimdVecTest, testImax)
    {
        using Scalar = TypeParam;
        using SimdType = SimdVec<Scalar>;
        size_t constexpr SS = SimdType::size();

        SimdIndex<Scalar> const idx = indexSequence<Scalar>();

        for (int i = 0; i < 100; ++i)
        {
            blaze::StaticVector<Scalar, SS> a;
            randomize(a);
            SimdVec<Scalar> const v {a.data(), false};

            SimdVec<Scalar> v_max;
            SimdIndex<Scalar> idx_max;
            std::tie(v_max, idx_max) = imax(v, idx);

            auto const max_el = std::max_element(a.begin(), a.end());
            for (size_t i = 0; i < idx_max.size; ++i)
                ASSERT_EQ(idx_max.get(i), std::distance(a.begin(), max_el)) << "Error at element " << i;

            for (size_t i = 0; i < v_max.size(); ++i)
                ASSERT_EQ(v_max[i], *max_el) << "Error at element " << i;
        }
    }
}
