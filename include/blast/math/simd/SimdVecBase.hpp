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

#pragma once

#include <blast/math/simd/SimdSize.hpp>

#include <xsimd/xsimd.hpp>

#include <cstdlib>


namespace blast
{
    template <typename T, typename Arch>
    class SimdVecBase
    {
        using XSimdType = xsimd::batch<T, Arch>;

    public:
        using ValueType = T;
        using IntrinsicType = typename XSimdType::register_type;


        /**
         * @brief Number of elements in SIMD pack
         */
        static size_t constexpr size()
        {
            return XSimdType::size;
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
        }

    protected:
        SimdVecBase(IntrinsicType value)
        :   value_ {value}
        {
        }

        XSimdType value_;
    };
}
