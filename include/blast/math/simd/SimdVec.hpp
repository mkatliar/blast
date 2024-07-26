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

#include <blast/math/simd/SimdVecBase.hpp>

#include <xsimd/xsimd.hpp>

#include <tuple>


namespace blast
{
    template <typename T, typename Arch = xsimd::default_arch>
    class SimdVec;


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
    typename SimdVec<T, Arch>::MaskType operator>(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept;

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
    :   public SimdVecBase<T, Arch>
    {
    public:
        using typename SimdVecBase<T, Arch>::ValueType;
        using typename SimdVecBase<T, Arch>::IntrinsicType;
        using typename SimdVecBase<T, Arch>::XSimdType;
        using MaskType = xsimd::batch_bool<T, Arch>;


        /**
         * @brief Set to [0, 0, 0, ...]
         */
        SimdVec() noexcept
        :   SimdVecBase<T, Arch> {T {}}
        {
        }


        SimdVec(SimdVec const&) noexcept = default;


        SimdVec(IntrinsicType value) noexcept
        :   SimdVecBase<T, Arch> {value}
        {
        }


        SimdVec(XSimdType value) noexcept
        :   SimdVecBase<T, Arch> {value}
        {
        }


        /**
         * @brief Set to [value, value, ...]
         *
         * @param value value for each component of SIMD vector
         */
        SimdVec(ValueType value) noexcept
        :   SimdVecBase<T, Arch> {value}
        {
        }


        /**
         * @brief Load from location
         *
         * @param src memory location to load from
         * @param aligned true indicates that an aligned read instruction should be used
         */
        explicit SimdVec(ValueType const * src, bool aligned) noexcept
        :   SimdVecBase<T, Arch> {aligned ? xsimd::load_aligned(src) : xsimd::load_unaligned(src)}
        {
        }


        /**
         * @brief Masked load from location
         *
         * @param src memory location to load from
         * @param mask load mask
         * @param aligned true if @a src is SIMD-aligned
         */
        explicit SimdVec(ValueType const * src, MaskType mask, bool aligned) noexcept;


        /**
         * @brief Store to memory
         *
         * @param dst memory location to store to
         * @param aligned true if @a dst is SIMD-aligned
         */
        void store(ValueType * dst, bool aligned) const noexcept
        {
            if (aligned)
                xsimd::store_aligned(dst, this->value_);
            else
                xsimd::store_unaligned(dst, this->value_);
        }


        /**
         * @brief Masked store to memory
         *
         * @param dst memory location to store to
         * @param mask store mask
         * @param aligned true if @a dst is SIMD-aligned
         */
        void store(ValueType * dst, MaskType mask, bool aligned) const noexcept;


        /**
         * @brief In-place multiplication
         *
         * @param a multiplier
         *
         * @return @a *this after multiplication with @a a
         */
        SimdVec& operator*=(SimdVec const& a) noexcept
        {
            this->value_ *= a.value_;
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
            this->value_ /= a.value_;
            return *this;
        }


        friend SimdVec fmadd<>(SimdVec const& a, SimdVec const& b, SimdVec const& c) noexcept;
        friend SimdVec blend<>(SimdVec const& a, SimdVec const& b, MaskType mask) noexcept;
        friend SimdVec abs<>(SimdVec const& a) noexcept;
        friend SimdVec max<>(SimdVec const& a, SimdVec const& b) noexcept;
        friend ValueType max<>(SimdVec const& x) noexcept;
        friend MaskType operator><>(SimdVec const& a, SimdVec const& b) noexcept;
        friend SimdVec operator*<>(SimdVec const& a, SimdVec const& b) noexcept;
        friend SimdVec operator*<>(ValueType const& a, SimdVec const& b) noexcept;
        friend SimdVec operator*<>(SimdVec const& a, ValueType const& b) noexcept;

        template <typename Index>
        friend std::tuple<SimdVec, SimdVec<Index, xsimd::avx2>> imax(SimdVec const& v1, SimdVec<Index, xsimd::avx2> const& idx);
    };


    template <typename T, typename Arch>
    inline SimdVec<T, Arch> fmadd(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b, SimdVec<T, Arch> const& c) noexcept
    {
        return xsimd::fma(a.value_, b.value_, c.value_);
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
    inline typename SimdVec<T, Arch>::MaskType operator>(SimdVec<T, Arch> const& a, SimdVec<T, Arch> const& b) noexcept
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
