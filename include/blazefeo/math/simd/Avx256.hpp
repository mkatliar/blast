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

#include <cstddef>
#include <immintrin.h>

#include <cstdint>


namespace blazefeo
{
    template <typename T>
    class SimdPack;


    template <>
    class SimdPack<double>
    {
    public:
        using ValueType = double;
        using IntrinsicType = __m256d;
        using MaskType = __m256i;

        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdPack() noexcept
        :   value_ {_mm256_setzero_pd()}
        {
        }


        SimdPack(IntrinsicType value) noexcept
        :   value_ {value}
        {
        }


        SimdPack(ValueType value) noexcept
        :   value_ {_mm256_set1_pd(value)}
        {
        }


        SimdPack(ValueType const * src, bool aligned) noexcept
        :   value_ {aligned ? _mm256_load_pd(src) : _mm256_loadu_pd(src)}
        {
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
        }


        friend MaskType operator>(SimdPack const& a, SimdPack const& b) noexcept
        {
            return _mm256_cmp_pd(a.value_, b.value_, _CMP_GT_OQ);
        }


        friend SimdPack blend(SimdPack const& a, SimdPack const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_pd(a.value_, b.value_, mask);
        }


        friend SimdPack abs(SimdPack const& a) noexcept
        {
            return _mm256_andnot_pd(SimdPack {-0.}, a.value_);
        }


        /**
        * @brief Maximum across all alements
        *
        * The implementation is based on
        * https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
        *
        * @return max(x[0], x[1], x[2], x[3])
        */
        friend ValueType max(SimdPack x) noexcept
        {
            __m256d y = _mm256_permute2f128_pd(x.value_, x.value_, 1); // permute 128-bit values
            __m256d m1 = _mm256_max_pd(x.value_, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
            __m256d m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
            __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])

            return m[0];
        }


        /**
         * @brief Number of elements in SIMD pack
         */
        static size_t constexpr size()
        {
            return 4;
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


    template <>
    class SimdPack<float>
    {
    public:
        using ValueType = float;
        using IntrinsicType = __m256;
        using MaskType = __m256i;

        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdPack() noexcept
        :   value_ {_mm256_setzero_ps()}
        {
        }


        SimdPack(IntrinsicType value) noexcept
        :   value_ {value}
        {
        }


        SimdPack(ValueType value) noexcept
        :   value_ {_mm256_set1_ps(value)}
        {
        }


        SimdPack(ValueType const * src, bool aligned) noexcept
        :   value_ {aligned ? _mm256_load_ps(src) : _mm256_loadu_ps(src)}
        {
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
        }


        friend MaskType operator>(SimdPack const& a, SimdPack const& b) noexcept
        {
            return _mm256_cmp_pd(a.value_, b.value_, _CMP_GT_OQ);
        }


        friend SimdPack blend(SimdPack const& a, SimdPack const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_ps(a.value_, b.value_, mask);
        }


        friend SimdPack abs(SimdPack const& a) noexcept
        {
            return _mm256_andnot_ps(SimdPack {-0.f}, a.value_);
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
        friend ValueType max(SimdPack x)
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


    template <>
    class SimdPack<std::int64_t>
    {
    public:
        using ValueType = std::int64_t;
        using IntrinsicType = __m256i;
        using MaskType = __m256i;

        /**
         * @brief Initialize to [0, 0, 0, 0]
         */
        SimdPack() noexcept
        :   value_ {_mm256_setzero_si256()}
        {
        }


        /**
         * @brief Initialize to [a, a, a, a]
         */
        SimdPack(ValueType a) noexcept
        :   value_ {_mm256_set1_epi64x(a)}
        {
        }


        /**
         * @brief Initialize to [a3, a2, a1, a0]
         */
        SimdPack(ValueType a3, ValueType a2, ValueType a1, ValueType a0) noexcept
        :   value_ {_mm256_set_epi64x(a3, a2, a1, a0)}
        {
        }


        SimdPack(IntrinsicType value) noexcept
        :   value_ {value}
        {
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
        }


        friend MaskType operator>(SimdPack const& a, SimdPack const& b) noexcept
        {
            return _mm256_cmpgt_epi64(a.value_, b.value_);
        }


        friend SimdPack blend(SimdPack const& a, SimdPack const& b, MaskType mask) noexcept
        {
            return _mm256_blendv_epi8(a.value_, b.value_, mask);
        }


        friend SimdPack operator+(SimdPack const& a, ValueType n) noexcept
        {
            return _mm256_add_epi64(a.value_, _mm256_set1_epi64x(n));
        }


        /**
         * @brief Number of elements in SIMD pack
         */
        static size_t constexpr size()
        {
            return 4;
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