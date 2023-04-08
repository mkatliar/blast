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
#include <blast/math/simd/avx256/SimdSize.hpp>

#include <immintrin.h>

#include <tuple>


namespace blast
{
    template <>
    class SimdVec<float>
    :   public SimdVecBase<float, __m256>
    {
    public:
        using MaskType = __m256i;

        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdVec() noexcept
        :   SimdVecBase {_mm256_setzero_ps()}
        {
        }


        SimdVec(IntrinsicType value) noexcept
        :   SimdVecBase {value}
        {
        }


        SimdVec(ValueType value) noexcept
        :   SimdVecBase {_mm256_set1_ps(value)}
        {
        }


        /**
         * @brief Load from location
         *
         * @param src memory location to load from
         * @param aligned true if @a src is SIMD-aligned
         */
        SimdVec(ValueType const * src, bool aligned) noexcept
        :   SimdVecBase {aligned ? _mm256_load_ps(src) : _mm256_loadu_ps(src)}
        {
        }


        /**
         * @brief Masked load from location
         *
         * @param src memory location to load from
         * @param mask load mask
         * @param aligned true if @a src is SIMD-aligned
         */
        SimdVec(ValueType const * src, MaskType mask, bool aligned) noexcept
        :   SimdVecBase {_mm256_maskload_ps(src, mask)}
        {
        }


        /**
         * @brief Store to memory
         *
         * @param dst memory location to store to
         * @param aligned true if @a dst is SIMD-aligned
         */
        void store(ValueType * dst, bool aligned) const noexcept
        {
            if (aligned)
                _mm256_store_ps(dst, value_);
            else
                _mm256_storeu_ps(dst, value_);
        }


        /**
         * @brief Masked store to memory
         *
         * @param dst memory location to store to
         * @param mask store mask
         * @param aligned true if @a dst is SIMD-aligned
         */
        void store(ValueType * dst, MaskType mask, bool aligned) const noexcept
        {
            _mm256_maskstore_ps(dst, mask, value_);
        }


        friend MaskType operator>(SimdVec const& a, SimdVec const& b) noexcept
        {
            return _mm256_castps_si256(_mm256_cmp_ps(a.value_, b.value_, _CMP_GT_OQ));
        }


        friend SimdVec operator*(ValueType a, SimdVec const& x) noexcept
        {
            return a * x.value_;
        }


        friend SimdVec blend(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_ps(a.value_, b.value_, _mm256_castsi256_ps(mask));
        }


        friend SimdVec abs(SimdVec const& a) noexcept
        {
            return _mm256_andnot_ps(SimdVec {-0.f}, a.value_);
        }


        /**
        * @brief Vertical max (across two vectors)
        *
        * @param a first vector
        * @param b second vector
        *
        * @return [max(a[7], b7), [max(a[6], b6), max(a[5], b5), max(a[4], b4), max(a[3], b3), max(a[2], b2), max(a[1], b1), max(a[0], b0)]
        */
        friend SimdVec max(SimdVec const& a, SimdVec const& b)
        {
            return _mm256_max_ps(a, b);
        }


        /**
        * @brief Horizontal max (across all elements)
        *
        * The implementation is based on
        * https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
        *
        * @param a pack of 32-bit floating point numbers
        *
        * @return max(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
        */
        friend ValueType max(SimdVec const& x)
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


        template <typename Index>
        requires (SimdSize_v<Index> == size())
        friend std::tuple<SimdVec, SimdVec<Index>> imax(SimdVec const& v1, SimdVec<Index> const& idx)
        {
            /* v2 = [G H E F | C D A B]                                                                         */
            SimdVec const v2 = _mm256_permute_ps(v1, 0b10'11'00'01);
            SimdVec<Index> const iv2 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(idx), 0b10'11'00'01));

            /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */
            /* v3 = [W W Z Z | Y Y X X]                                                                         */
            // __m256 v3 = _mm256_max_ps(v1, v2);
            MaskType const mask_v3 = v2 > v1;
            SimdVec const v3 = blend(v1, v2, mask_v3);
            SimdVec<Index> const iv3 = blend(idx, iv2, mask_v3);

            /* v4 = [Z Z W W | X X Y Y]                                                                         */
            SimdVec const v4 = _mm256_permute_ps(v3, 0b00'00'10'10);
            SimdVec<Index> const iv4 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(iv3), 0b00'00'10'10));

            /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */
            /* v5 = [J J J J | I I I I]                                                                         */
            // __m256 v5 = _mm256_max_ps(v3, v4);
            MaskType const mask_v5 = v4 > v3;
            SimdVec const v5 = blend(v3, v4, mask_v5);
            SimdVec<Index> const iv5 = blend(iv3, iv4, mask_v5);

            /* v6 = [I I I I | J J J J]                                                                         */
            SimdVec const v6 = _mm256_permute2f128_ps(v5, v5, 0b0000'0001);
            SimdVec<Index> const iv6 = _mm256_castps_si256(
                _mm256_permute2f128_ps(
                    _mm256_castsi256_ps(iv5),
                    _mm256_castsi256_ps(iv5),
                    0b0000'0001
                )
            );

            /* v7 = [M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J) | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)] */
            // __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6);
            MaskType const mask_v7 = v6 > v5;
            SimdVec const v7 = blend(v5, v6, mask_v7);
            SimdVec<Index> const iv7 = blend(iv5, iv6, mask_v7);

            return std::make_tuple(v7, iv7);
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