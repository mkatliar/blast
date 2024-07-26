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
    inline SimdVec<float, xsimd::avx2>::SimdVec() noexcept
    :   SimdVecBase {_mm256_setzero_ps()}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(IntrinsicType value) noexcept
    :   SimdVecBase {value}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(ValueType value) noexcept
    :   SimdVecBase {_mm256_set1_ps(value)}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(ValueType const * src, bool aligned) noexcept
    :   SimdVecBase {aligned ? _mm256_load_ps(src) : _mm256_loadu_ps(src)}
    {
    }


    template <>
    inline SimdVec<float, xsimd::avx2>::SimdVec(ValueType const * src, MaskType mask, bool aligned) noexcept
    :   SimdVecBase {_mm256_maskload_ps(src, mask)}
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
    inline void SimdVec<float, xsimd::avx2>::store(ValueType * dst, MaskType mask, bool aligned) const noexcept
    {
        _mm256_maskstore_ps(dst, mask, value_);
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


    /*
    NOTE: without this specialization some tests fail, e.g.:

    [ RUN      ] RegisterMatrixTest/5.testPartialGerNT @ ./test/blast/math/simd/RegisterMatrixTest.cpp:386
    │ Failure @ ./test/blast/math/simd/RegisterMatrixTest.cpp:420
    │ │ Expected equality of these values:
    │ │   ker(i, j)
    │ │     Which is: 0.482636
    │ │   i < m && j < n ? D_ref(i, j) : 0.
    │ │     Which is: 0.482636
    │ element mismatch at (0, 1), store size = 1x2
    [  FAILED  ] RegisterMatrixTest/5.testPartialGerNT, where TypeParam = blast::RegisterMatrix<float, 8ul, 4ul, true> (1 ms)
    */
    template <>
    inline SimdVec<float, xsimd::avx2> fmadd(SimdVec<float, xsimd::avx2> const& a, SimdVec<float, xsimd::avx2> const& b, SimdVec<float, xsimd::avx2> const& c) noexcept
    {
        return _mm256_fmadd_ps(a.value_, b.value_, c.value_);
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


    template <typename Index>
    requires (SimdSize_v<Index> == SimdVec<float, xsimd::avx2>::size())
    inline std::tuple<SimdVec<float, xsimd::avx2>, SimdVec<Index, xsimd::avx2>> imax(SimdVec<float, xsimd::avx2> const& v1, SimdVec<Index, xsimd::avx2> const& idx) noexcept
    {
        /* v2 = [G H E F | C D A B]                                                                         */
        SimdVec<float, xsimd::avx2> const v2 = _mm256_permute_ps(v1, 0b10'11'00'01);
        SimdVec<Index, xsimd::avx2> const iv2 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(idx), 0b10'11'00'01));

        /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */
        /* v3 = [W W Z Z | Y Y X X]                                                                         */
        // __m256 v3 = _mm256_max_ps(v1, v2);
        SimdVec<float, xsimd::avx2>::MaskType const mask_v3 = v2 > v1;
        SimdVec<float, xsimd::avx2> const v3 = blend(v1, v2, mask_v3);
        SimdVec<Index, xsimd::avx2> const iv3 = blend(idx, iv2, mask_v3);

        /* v4 = [Z Z W W | X X Y Y]                                                                         */
        SimdVec<float, xsimd::avx2> const v4 = _mm256_permute_ps(v3, 0b00'00'10'10);
        SimdVec<Index, xsimd::avx2> const iv4 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(iv3), 0b00'00'10'10));

        /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */
        /* v5 = [J J J J | I I I I]                                                                         */
        // __m256 v5 = _mm256_max_ps(v3, v4);
        SimdVec<float, xsimd::avx2>::MaskType const mask_v5 = v4 > v3;
        SimdVec<float, xsimd::avx2> const v5 = blend(v3, v4, mask_v5);
        SimdVec<Index, xsimd::avx2> const iv5 = blend(iv3, iv4, mask_v5);

        /* v6 = [I I I I | J J J J]                                                                         */
        SimdVec<float, xsimd::avx2> const v6 = _mm256_permute2f128_ps(v5, v5, 0b0000'0001);
        SimdVec<Index, xsimd::avx2> const iv6 = _mm256_castps_si256(
            _mm256_permute2f128_ps(
                _mm256_castsi256_ps(iv5),
                _mm256_castsi256_ps(iv5),
                0b0000'0001
            )
        );

        /* v7 = [M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J) | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)] */
        // __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6);
        SimdVec<float, xsimd::avx2>::MaskType const mask_v7 = v6 > v5;
        SimdVec<float, xsimd::avx2> const v7 = blend(v5, v6, mask_v7);
        SimdVec<Index, xsimd::avx2> const iv7 = blend(iv5, iv6, mask_v7);

        return std::make_tuple(v7, iv7);
    }
}
