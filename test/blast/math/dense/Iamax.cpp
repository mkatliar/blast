// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <blast/math/dense/Iamax.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


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
            DynamicVector<Real> x(n);
            randomize(x);

            size_t const ind = iamax(x);

            for (auto v : x)
                ASSERT_LE(std::abs(v), x[ind]) << "Error at size " << n;
        }
    }
}