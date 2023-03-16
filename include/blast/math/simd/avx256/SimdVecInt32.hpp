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

#include <blast/math/simd/SimdVec.hpp>
#include <blast/math/simd/SimdVecBase.hpp>
#include <blast/math/simd/SequenceTag.hpp>
#include <blast/math/simd/avx256/SimdSize.hpp>

#include <immintrin.h>

#include <cstdint>


namespace blast
{
    template <>
    class SimdVec<std::int32_t>
    :   public SimdVecBase<std::int32_t, __m256i>
    {
    public:
        using MaskType = __m256i;

        /**
         * @brief Initialize to (0, 0, 0, 0, 0, 0, 0, 0)
         */
        SimdVec() noexcept
        :   SimdVecBase {_mm256_setzero_si256()}
        {
        }


        /**
         * @brief Initialize to (a, a, a, a, a, a, a, a)
         */
        SimdVec(ValueType a) noexcept
        :   SimdVecBase {_mm256_set1_epi32(a)}
        {
        }


        /**
         * @brief Initialize to (n + 7, n + 6, n + 5, n + 4, n + 3, n + 2, n + 1, n)
         *
         * @param n
         */
        SimdVec(simd::SequenceTag, ValueType n = 0)
        :   SimdVec {n + 7, n + 6, n + 5, n + 4, n + 3, n + 2, n + 1, n}
        {
        }


        /**
         * @brief Initialize to (a7, a6, a5, a4, a3, a2, a1, a0),
         * where a0 corresponds to the lower bits.
         */
        SimdVec(ValueType a7, ValueType a6, ValueType a5, ValueType a4,
            ValueType a3, ValueType a2, ValueType a1, ValueType a0) noexcept
        :   SimdVecBase {_mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0)}
        {
        }


        SimdVec(IntrinsicType value) noexcept
        :   SimdVecBase {value}
        {
        }


        SimdVec& operator+=(ValueType x) noexcept
        {
            value_ = _mm256_add_epi32(value_, _mm256_set1_epi32(x));
            return *this;
        }


        friend MaskType operator>(SimdVec const& a, SimdVec const& b) noexcept
        {
            return _mm256_cmpgt_epi32(a.value_, b.value_);
        }


        friend SimdVec blend(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_epi8(a.value_, b.value_, mask);
        }


        friend SimdVec operator+(SimdVec const& a, ValueType n) noexcept
        {
            return _mm256_add_epi32(a.value_, _mm256_set1_epi32(n));
        }


        /**
         * @brief Access single element
         *
         * @param i element index
         *
         * @return element value
         */
        ValueType operator[](size_t i) const noexcept
        {
            alignas(IntrinsicType) ValueType v[size()];
            _mm256_storeu_si256(reinterpret_cast<IntrinsicType *>(v), value_);

            return v[i];
        }
    };
}