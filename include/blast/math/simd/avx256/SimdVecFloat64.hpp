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
    inline SimdVec<double, xsimd::avx2>::SimdVec() noexcept
    :   value_ {_mm256_setzero_pd()}
    {
    }


    template <>
    inline SimdVec<double, xsimd::avx2>::SimdVec(IntrinsicType value) noexcept
    :   value_ {value}
    {
    }


    template <>
    inline SimdVec<double, xsimd::avx2>::SimdVec(ValueType value) noexcept
    :   value_ {_mm256_set1_pd(value)}
    {
    }


    template <>
    inline SimdVec<double, xsimd::avx2>::SimdVec(ValueType const * src, bool aligned) noexcept
    :   value_ {aligned ? _mm256_load_pd(src) : _mm256_loadu_pd(src)}
    {
    }


    template <>
    inline void SimdVec<double, xsimd::avx2>::store(ValueType * dst, bool aligned) const noexcept
    {
        if (aligned)
            _mm256_store_pd(dst, value_);
        else
            _mm256_storeu_pd(dst, value_);
    }


    template <>
    inline SimdVec<double, xsimd::avx2>::MaskType operator>(SimdVec<double, xsimd::avx2> const& a, SimdVec<double, xsimd::avx2> const& b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a.value_, b.value_, _CMP_GT_OQ));
    }


    template <>
    inline SimdVec<double, xsimd::avx2> operator*(SimdVec<double, xsimd::avx2> const& a, SimdVec<double, xsimd::avx2> const& b) noexcept
    {
        return _mm256_mul_pd(a.value_, b.value_);
    }


    template <>
    inline SimdVec<double, xsimd::avx2> fmadd(SimdVec<double, xsimd::avx2> const& a, SimdVec<double, xsimd::avx2> const& b, SimdVec<double, xsimd::avx2> const& c) noexcept
    {
        return _mm256_fmadd_pd(a.value_, b.value_, c.value_);
    }


    template <>
    inline SimdVec<double, xsimd::avx2> blend(SimdVec<double, xsimd::avx2> const& a, SimdVec<double, xsimd::avx2> const& b, SimdVec<double, xsimd::avx2>::MaskType mask) noexcept
    {
        return _mm256_blendv_pd(a.value_, b.value_, _mm256_castsi256_pd(mask));
    }


    template <>
    inline SimdVec<double, xsimd::avx2> abs(SimdVec<double, xsimd::avx2> const& a) noexcept
    {
        return _mm256_andnot_pd(_mm256_set1_pd(-0.), a.value_);
    }


    template <>
    inline SimdVec<double, xsimd::avx2> max(SimdVec<double, xsimd::avx2> const& a, SimdVec<double, xsimd::avx2> const& b) noexcept
    {
        return _mm256_max_pd(a, b);
    }


    template <>
    inline double max(SimdVec<double, xsimd::avx2> const& x) noexcept
    {
        __m256d y = _mm256_permute2f128_pd(x.value_, x.value_, 1); // permute 128-bit values
        __m256d m1 = _mm256_max_pd(x.value_, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        __m256d m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
        __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])

        return m[0];
    }
}
