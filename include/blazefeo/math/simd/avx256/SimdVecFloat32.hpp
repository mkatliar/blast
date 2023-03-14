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

#include <immintrin.h>


namespace blazefeo
{
    template <>
    class SimdVec<float>
    {
    public:
        using ValueType = float;
        using IntrinsicType = __m256;
        using MaskType = __m256i;

        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdVec() noexcept
        :   value_ {_mm256_setzero_ps()}
        {
        }


        SimdVec(IntrinsicType value) noexcept
        :   value_ {value}
        {
        }


        SimdVec(ValueType value) noexcept
        :   value_ {_mm256_set1_ps(value)}
        {
        }


        SimdVec(ValueType const * src, bool aligned) noexcept
        :   value_ {aligned ? _mm256_load_ps(src) : _mm256_loadu_ps(src)}
        {
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
        }


        friend MaskType operator>(SimdVec const& a, SimdVec const& b) noexcept
        {
            return _mm256_cmp_ps(a.value_, b.value_, _CMP_GT_OQ);
        }


        friend SimdVec blend(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_ps(a.value_, b.value_, mask);
        }


        friend SimdVec abs(SimdVec const& a) noexcept
        {
            return _mm256_andnot_ps(SimdVec {-0.f}, a.value_);
        }


        /**
        * @brief Max across all elements
        *
        * The implementation is based on
        * https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
        *
        * @param a pack of 32-bit floating point numbers
        *
        * @return max(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
        */
        friend ValueType max(SimdVec x)
        {
            __m256 v1 = x.value_;                                          /* v1 = [H G F E | D C B A]                                                                         */
            __m256 v2 = _mm256_permute_ps(v1, 0b10'11'00'01);       /* v2 = [G H E F | C D A B]                                                                         */
            __m256 v3 = _mm256_max_ps(v1, v2);                      /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */
                                                                    /* v3 = [W W Z Z | Y Y X X]                                                                         */
            __m256 v4 = _mm256_permute_ps(v3, 0b00'00'10'10);       /* v4 = [Z Z W W | X X Y Y]                                                                         */
            __m256 v5 = _mm256_max_ps(v3, v4);                      /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */
                                                                    /* v5 = [J J J J | I I I I]                                                                         */
            __m128 v6 = _mm256_extractf128_ps(v5, 1);               /* v6 = [- - - - | J J J J]                                                                         */
            __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6); /* v7 = [- - - - | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)]                                     */

            return v7[0];
        }


        /**
         * @brief Number of elements in SIMD pack
         */
        static size_t constexpr size()
        {
            return 8;
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


    private:
        IntrinsicType value_;
    };
}