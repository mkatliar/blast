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

#include <blazefeo/math/simd/SimdVec.hpp>
#include <blazefeo/math/simd/SimdVecBase.hpp>

#include <immintrin.h>


namespace blazefeo
{
    template <>
    class SimdVec<double>
    :   public SimdVecBase<double, __m256d>
    {
    public:
        using MaskType = __m256i;

        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdVec() noexcept
        :   SimdVecBase {_mm256_setzero_pd()}
        {
        }


        SimdVec(IntrinsicType value) noexcept
        :   SimdVecBase {value}
        {
        }


        SimdVec(ValueType value) noexcept
        :   SimdVecBase {_mm256_set1_pd(value)}
        {
        }


        SimdVec(ValueType const * src, bool aligned) noexcept
        :   SimdVecBase {aligned ? _mm256_load_pd(src) : _mm256_loadu_pd(src)}
        {
        }


        friend MaskType operator>(SimdVec const& a, SimdVec const& b) noexcept
        {
            return _mm256_cmp_pd(a.value_, b.value_, _CMP_GT_OQ);
        }


        friend SimdVec blend(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_pd(a.value_, b.value_, mask);
        }


        friend SimdVec abs(SimdVec const& a) noexcept
        {
            return _mm256_andnot_pd(SimdVec {-0.}, a.value_);
        }


        /**
        * @brief Maximum across all alements
        *
        * The implementation is based on
        * https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
        *
        * @return max(x[0], x[1], x[2], x[3])
        */
        friend ValueType max(SimdVec x) noexcept
        {
            __m256d y = _mm256_permute2f128_pd(x.value_, x.value_, 1); // permute 128-bit values
            __m256d m1 = _mm256_max_pd(x.value_, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
            __m256d m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
            __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])

            return m[0];
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
            return value_[i];
        }
    };
}