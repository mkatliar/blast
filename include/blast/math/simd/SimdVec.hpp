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

#include <blast/math/simd/SimdIndex.hpp>
#include <blast/math/simd/SimdMask.hpp>

#include <xsimd/xsimd.hpp>

#include <tuple>


namespace blast
{
    template <typename T, typename Arch = xsimd::default_arch>
    class SimdVec;


    template <typename Arch, typename T>
    xsimd::batch<T, Arch> maskload(T const * src, xsimd::batch_bool<T, Arch> const& mask) noexcept;

    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline xsimd::batch<float, Arch> maskload(float const * src, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        return _mm256_maskload_ps(src, mask);
    }

    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline xsimd::batch<double, Arch> maskload(double const * src, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        return _mm256_maskload_pd(src, mask);
    }

    template <typename Arch, typename T>
    void maskstore(xsimd::batch<T, Arch> const& v, T * dst, xsimd::batch_bool<T, Arch> const& mask) noexcept;

    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline void maskstore(xsimd::batch<float, Arch> const& v, float * dst, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        _mm256_maskstore_ps(dst, xsimd::batch_bool_cast<std::int32_t>(mask), v);
    }

    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline void maskstore(xsimd::batch<double, Arch> const& v, double * dst, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        _mm256_maskstore_pd(dst, xsimd::batch_bool_cast<std::int64_t>(mask), v);
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline std::tuple<xsimd::batch<float, Arch>, xsimd::batch<std::int32_t, Arch>> imax(xsimd::batch<float, Arch> const& v1, xsimd::batch<std::int32_t, Arch> const& idx) noexcept
    {
        /* v2 = [G H E F | C D A B]                                                                         */
        __m256 const v2 = _mm256_permute_ps(v1, 0b10'11'00'01);
        __m256i const iv2 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(idx), 0b10'11'00'01));

        /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */
        /* v3 = [W W Z Z | Y Y X X]                                                                         */
        // __m256 v3 = _mm256_max_ps(v1, v2);
        __m256 const mask_v3 = _mm256_cmp_ps(v2, v1, _CMP_GT_OQ);
        __m256 const v3 = _mm256_blendv_ps(v1, v2, mask_v3);
        __m256i const iv3 = _mm256_blendv_epi8(idx, iv2, _mm256_castps_si256(mask_v3));

        /* v4 = [Z Z W W | X X Y Y]                                                                         */
        __m256 const v4 = _mm256_permute_ps(v3, 0b00'00'10'10);
        __m256i const iv4 = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(iv3), 0b00'00'10'10));

        /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */
        /* v5 = [J J J J | I I I I]                                                                         */
        // __m256 v5 = _mm256_max_ps(v3, v4);
        __m256 const mask_v5 = _mm256_cmp_ps(v4, v3, _CMP_GT_OQ);
        __m256 const v5 = _mm256_blendv_ps(v3, v4, mask_v5);
        __m256i const iv5 = _mm256_blendv_epi8(iv3, iv4, _mm256_castps_si256(mask_v5));

        /* v6 = [I I I I | J J J J]                                                                         */
        __m256 const v6 = _mm256_permute2f128_ps(v5, v5, 0b0000'0001);
        __m256i const iv6 = _mm256_castps_si256(
            _mm256_permute2f128_ps(
                _mm256_castsi256_ps(iv5),
                _mm256_castsi256_ps(iv5),
                0b0000'0001
            )
        );

        /* v7 = [M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J) | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)] */
        // __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6);
        __m256 const mask_v7 = _mm256_cmp_ps(v6, v5, _CMP_GT_OQ);
        __m256 const v7 = _mm256_blendv_ps(v5, v6, mask_v7);
        __m256i const iv7 = _mm256_blendv_epi8(iv5, iv6, _mm256_castps_si256(mask_v7));

        return {v7, iv7};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::avx2, Arch>
    inline std::tuple<xsimd::batch<double, Arch>, xsimd::batch<std::int64_t, Arch>> imax(xsimd::batch<double, Arch> const& x, xsimd::batch<std::int64_t, Arch> const& idx) noexcept
    {
        __m256d const y = _mm256_permute2f128_pd(x, x, 1); // permute 128-bit values
        __m256i const iy = _mm256_permute2f128_si256(idx, idx, 1);

        // __m256d m1 = _mm256_max_pd(x.value_, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        __m256d const mask_m1 = _mm256_cmp_pd(y, x, _CMP_GT_OQ);
        __m256d const m1 = _mm256_blendv_pd(x, y, mask_m1);
        __m256i const im1 = _mm256_blendv_epi8(idx, iy, _mm256_castpd_si256(mask_m1));

        __m256d const m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
        __m256i const im2 = _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(im1), 5));

        // __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])
        __m256d const mask_m = _mm256_cmp_pd(m2, m1, _CMP_GT_OQ);
        __m256d const m = _mm256_blendv_pd(m1, m2, mask_m);
        __m256i const im = _mm256_blendv_epi8(im1, im2, _mm256_castpd_si256(mask_m));

        return {m, im};
    }


    /**
    * @brief Fused negative multiply-add
    *
    * Calculate -a * b + c
    *
    * @param a first multiplier
    * @param b second multiplier
    * @param c addendum
    *
    * @return @a a * @a b + @a c element-wise
    */
    template <typename T, typename Arch>
    SimdVec<T, Arch> fnmadd(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, SimdVec<T, Arch> const& c) noexcept;

    /**
    * @brief Fused multiply-add
    *
    * Calculate a * b + c
    *
    * @param a first multiplier
    * @param b second multiplier
    * @param c addendum
    *
    * @return @a a * @a b + @a c element-wise
    */
    template <typename T, typename Arch>
    SimdVec<T, Arch> fmadd(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, SimdVec<T, Arch> const& c) noexcept;

    template <typename T, typename Arch>
    SimdVec<T, Arch> blend(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, typename SimdVec<T, Arch>::MaskType mask) noexcept;

    template <typename T, typename Arch>
    SimdVec<T, Arch> abs(SimdVec<T, Arch> const& a) noexcept;

    /**
    * @brief Vertical max (across two vectors)
    *
    * @param a first vector
    * @param b second vector
    *
    * @return [max(a[7], b7), [max(a[6], b6), max(a[5], b5), max(a[4], b4), max(a[3], b3), max(a[2], b2), max(a[1], b1), max(a[0], b0)]
    */
    template <typename T, typename Arch>
    SimdVec<T, Arch> max(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept;

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
    template <typename T, typename Arch>
    T max(SimdVec<T, Arch> const& x) noexcept;

    template <typename T, typename Arch>
    SimdMask<T, Arch> operator>(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept;

    /**
    * @brief Multiplication
    *
    * @param a first multiplier
    * @param b second multiplier
    *
    * @return product @a a * @a b
    */
    template <typename T, typename Arch>
    SimdVec<T, Arch> operator*(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept;

    /**
    * @brief Left multiplication with a scalar
    *
    * @param a scalar multiplier
    * @param b batch multiplier
    *
    * @return product @a a * @a b
    */
    template <typename T, typename Arch>
    SimdVec<T, Arch> operator*(T const& a, SimdVec<T, Arch> const& b) noexcept;


    /**
    * @brief Right multiplication with a scalar
    *
    * @param a batch multiplier
    * @param b scalar multiplier
    *
    * @return product @a a * @a b
    */
    template <typename T, typename Arch>
    SimdVec<T, Arch> operator*(SimdVec<T, Arch> const& a, T const& b) noexcept;


    /**
     * @brief Data-parallel type with a given element type.
     *
     * @tparam T element type
     */
    template <typename T, typename Arch>
    class SimdVec
    {
    public:
        using ValueType = T;
        using XSimdType = xsimd::batch<T, Arch>;
        using IntrinsicType = typename XSimdType::register_type;
        using MaskType = SimdMask<T, Arch>;


        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdVec() noexcept
        :   value_ {T {}}
        {
        }


        SimdVec(SimdVec const&) noexcept = default;


        SimdVec(IntrinsicType value) noexcept
        :   value_ {value}
        {
        }


        SimdVec(XSimdType value) noexcept
        :   value_ {value}
        {
        }


        /**
         * @brief Set to [value, value, ...]
         *
         * @param value value for each component of SIMD vector
         */
        SimdVec(ValueType value) noexcept
        :   value_ {value}
        {
        }


        /**
         * @brief Load from location
         *
         * @param src memory location to load from
         * @param aligned true indicates that an aligned read instruction should be used
         */
        explicit SimdVec(ValueType const * src, bool aligned) noexcept
        :   value_ {aligned ? xsimd::load_aligned(src) : xsimd::load_unaligned(src)}
        {
        }


        /**
         * @brief Masked load from location
         *
         * @param src memory location to load from
         * @param mask load mask
         * @param aligned true if @a src is SIMD-aligned
         */
        explicit SimdVec(ValueType const * src, MaskType mask, bool aligned) noexcept
        :   value_ {maskload(src, mask)}
        {
        }


        /**
         * @brief Number of elements in SIMD pack
         */
        static size_t constexpr size()
        {
            return XSimdType::size;
        }


        /**
         * @brief Set to 0
         */
        void reset() noexcept
        {
            value_ = ValueType {};
        }


        operator IntrinsicType() const noexcept
        {
            return value_;
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
            return value_.get(i);
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
                xsimd::store_aligned(dst, value_);
            else
                xsimd::store_unaligned(dst, value_);
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
            maskstore(value_, dst, mask);
        }


        /**
         * @brief In-place multiplication
         *
         * @param a multiplier
         *
         * @return @a *this after multiplication with @a a
         */
        SimdVec& operator*=(SimdVec const& a) noexcept
        {
            value_ *= a.value_;
            return *this;
        }


        /**
         * @brief In-place division
         *
         * @param a divisor
         *
         * @return @a *this after division by @a a
         */
        SimdVec& operator/=(SimdVec const& a) noexcept
        {
            value_ /= a.value_;
            return *this;
        }


        friend SimdVec fmadd<>(SimdVec const& a, SimdVec const& b, SimdVec const& c) noexcept;
        friend SimdVec fnmadd<>(SimdVec const& a, SimdVec const& b, SimdVec const& c) noexcept;
        friend SimdVec blend<>(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept;
        friend SimdVec abs<>(SimdVec const& a) noexcept;
        friend SimdVec max<>(SimdVec const& a, SimdVec const& b) noexcept;
        friend ValueType max<>(SimdVec const& x) noexcept;
        friend MaskType operator><>(SimdVec const& a, SimdVec const& b) noexcept;
        friend SimdVec operator*<>(SimdVec const& a, SimdVec const& b) noexcept;
        friend SimdVec operator*<>(ValueType const& a, SimdVec const& b) noexcept;
        friend SimdVec operator*<>(SimdVec const& a, ValueType const& b) noexcept;

        friend std::tuple<SimdVec, SimdIndex<T, Arch>> imax(SimdVec const& v1, SimdIndex<T, Arch> const& idx) noexcept
        {
            return imax(v1.value_, idx);
        }

    private:
        XSimdType value_;
    };


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> fmadd(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, SimdVec<T, Arch> const& c) noexcept
    {
        return xsimd::fma(a.value_, b.value_, c.value_);
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> fnmadd(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, SimdVec<T, Arch> const& c) noexcept
    {
        return xsimd::fnma(a.value_, b.value_, c.value_);
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> blend(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, typename SimdVec<T, Arch>::MaskType mask) noexcept
    {
        return xsimd::select(mask, a.value_, b.value_);
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> abs(SimdVec<T, Arch> const& a) noexcept
    {
        return xsimd::abs(a.value_);
    }


    template <typename T, typename Arch>
    inline SimdMask<T, Arch> operator>(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept
    {
        return xsimd::gt(a.value_, b.value_);
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> operator*(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept
    {
        return a.value_ * b.value_;
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> operator*(T const& a, SimdVec<T, Arch> const& b) noexcept
    {
        return SimdVec<T, Arch> {a} * b;
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> operator*(SimdVec<T, Arch> const& a, T const& b) noexcept
    {
        return a * SimdVec<T, Arch> {b};
    }


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> max(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept
    {
        return xsimd::max(a, b);
    }


    template <typename T, typename Arch>
    inline T max(SimdVec<T, Arch> const& x) noexcept
    {
        return xsimd::reduce_max(x.value_);
    }
}
