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


        /**
         * @brief Load from location
         *
         * @param src memory location to load from
         * @param aligned true if @a src is SIMD-aligned
         */
        SimdVec(ValueType const * src, bool aligned) noexcept
        :   SimdVecBase {aligned ? _mm256_load_pd(src) : _mm256_loadu_pd(src)}
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
        :   SimdVecBase {_mm256_maskload_pd(src, mask)}
        {
        }


        friend MaskType operator>(SimdVec const& a, SimdVec const& b) noexcept
        {
            return _mm256_castpd_si256(_mm256_cmp_pd(a.value_, b.value_, _CMP_GT_OQ));
        }


        friend SimdVec operator*(ValueType a, SimdVec const& x) noexcept
        {
            return a * x.value_;
        }


        friend SimdVec blend(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_pd(a.value_, b.value_, _mm256_castsi256_pd(mask));
        }


        friend SimdVec abs(SimdVec const& a) noexcept
        {
            return _mm256_andnot_pd(SimdVec {-0.}, a.value_);
        }


        /**
        * @brief Vertical max (across two vectors)
        *
        * @param a first vector
        * @param b second vector
        *
        * @return [max(a[3], b3), max(a[2], b2), max(a[1], b1), max(a[0], b0)]
        */
        friend SimdVec max(SimdVec const& a, SimdVec const& b)
        {
            return _mm256_max_pd(a, b);
        }


        /**
        * @brief Horizontal max (across all alements)
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


        template <typename Index>
        requires (SimdSize_v<Index> == size())
        friend std::tuple<SimdVec, SimdVec<Index>> imax(SimdVec const& x, SimdVec<Index> const& idx)
        {
            SimdVec const y = _mm256_permute2f128_pd(x.value_, x.value_, 1); // permute 128-bit values
            SimdVec<Index> const iy = _mm256_permute2f128_si256(idx, idx, 1);

            // __m256d m1 = _mm256_max_pd(x.value_, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
            MaskType const mask_m1 = y > x;
            SimdVec const m1 = blend(x, y, mask_m1);
            SimdVec<Index> const im1 = blend(idx, iy, mask_m1);

            SimdVec const m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
            SimdVec<Index> const im2 = _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(im1), 5));

            // __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
            MaskType const mask_m = m2 > m1;
            SimdVec const m = blend(m1, m2, mask_m);
            SimdVec<Index> const im = blend(im1, im2, mask_m);

            return std::make_tuple(m, im);
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