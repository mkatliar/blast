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

#include <immintrin.h>


namespace blast
{
    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec() noexcept
    :   value_ {_mm256_setzero_ps()}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(IntrinsicType value) noexcept
    :   value_ {value}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(ValueType value) noexcept
    :   value_ {_mm256_set1_ps(value)}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(ValueType const * src, bool aligned) noexcept
    :   value_ {aligned ? _mm256_load_ps(src) : _mm256_loadu_ps(src)}
    {
    }


    template <>
    inline void SimdVec<float, xsimd::avx2>::store(ValueType * dst, bool aligned) const noexcept
    {
        if (aligned)
            _mm256_store_ps(dst, value_);
        else
            _mm256_storeu_ps(dst, value_);
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::MaskType operator>(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a.value_, b.value_, _CMP_GT_OQ));
    }


    template <>
    inline SimdVec<float, xsimd::avx2> operator*(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b) noexcept
    {
        return _mm256_mul_ps(a.value_, b.value_);
    }


    template <>
    inline SimdVec<float, xsimd::avx2> fmadd(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b, SimdVec<float, xsimd::avx2> const& c) noexcept
    {
        return _mm256_fmadd_ps(a.value_, b.value_, c.value_);
    }


    template <>
    inline SimdVec<float, xsimd::avx2> fnmadd(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b, SimdVec<float, xsimd::avx2> const& c) noexcept
    {
        return _mm256_fnmadd_ps(a.value_, b.value_, c.value_);
    }


    template <>
    inline SimdVec<float, xsimd::avx2> blend(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b, SimdVec<float, xsimd::avx2>::MaskType mask) noexcept
    {
        return _mm256_blendv_ps(a.value_, b.value_, _mm256_castsi256_ps(mask));
    }


    template <>
    inline SimdVec<float, xsimd::avx2> abs(SimdVec<float, xsimd::avx2> const& a) noexcept
    {
        return _mm256_andnot_ps(SimdVec<float, xsimd::avx2> {-0.f}, a.value_);
    }


    template <>
    inline SimdVec<float, xsimd::avx2> max(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b) noexcept
    {
        return _mm256_max_ps(a, b);
    }


    template <>
    inline float max(SimdVec<float, xsimd::avx2> const& x) noexcept
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
}
