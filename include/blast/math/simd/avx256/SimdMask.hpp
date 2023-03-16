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

#include <immintrin.h>

#include <cstdint>


namespace blast
{
    template <typename T>
    class SimdMask
    {
        static size_t constexpr SS = SimdSize_v<T>;

    public:
        using ValueType = IntElementType_t<SS>;
        using IntrinsicType = __m256i;
        using MaskType = __m256i;


        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdMask() noexcept
        :   value_ {_mm256_setzero_pd()}
        {
        }


        SimdMask(IntrinsicType value) noexcept
        :   value_ {value}
        {
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
        }


        /**
         * @brief Number of elements in SIMD pack
         */
        static size_t constexpr size()
        {
            return SS;
        }


        /**
         * @brief Access specified element
         *
         * @param i element index
         *
         * @return element value
         */
        ValueType operator[](size_t i) const noexcept
        {
            return value_[i];
        }


    private:
        IntrinsicType value_;
    };
}