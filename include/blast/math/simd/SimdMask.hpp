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

#include <xsimd/xsimd.hpp>


namespace blast
{
    /**
     * @brief Data-parallel type with the element type bool.
     * The width of a given simd_mask instantiation is a constant expression, determined by the template parameter.
     *
     * @tparam T the element type simd_mask applies on
     * @tparam Arch instruction set architecture
     */
    template <typename T, typename Arch = xsimd::default_arch>
    class SimdMask
    :   public xsimd::batch_bool<T, Arch>
    {
    public:
        using XSimdType = xsimd::batch_bool<T, Arch>;
        using IntrinsicType = typename XSimdType::register_type;

        /**
         * @brief Construct from an intrinsic register type
         *
         * @param v the value to construct from
         */
        SimdMask(IntrinsicType v) noexcept
        :   XSimdType {v}
        {
        }

        /**
         * @brief Construct from an @a xsimd::batch_bool of the same type
         *
         * @param v the value to construct from
         */
        SimdMask(XSimdType const& v) noexcept
        :   XSimdType {v}
        {
        }

        /**
         * @brief Construct from an @a xsimd::batch_bool of a different type
         *
         * We need to allow conversion from batch_bool<U, Arch> with U != T,
         * because we need e.g. the following code to work:
         *
         * int n;
         * MaskType<double> mask = indexSequence<double>() < n;
         *
         * In the code above, the result of indexSequence<double>() is a batch<int64_t>,
         * and the result of indexSequence<double>() < n is a batch_bool<int64_t>,
         * which can not be directly assigned to a batch_bool<double>,
         * although their underlying register_type's are identical.
         *
         * @param v the value to construct from
         */
        template <typename U>
        SimdMask(xsimd::batch_bool<U, Arch> const& v) noexcept
        :   XSimdType {IntrinsicType(v)}
        {
        }

        /**
         * @brief In-place logical and.
         *
         * We need to define this operator to make the following code work:
         *
         * MaskType<double> mask;
         * int n;
         * mask &= indexSequence<double>() >= n;
         *
         * In the code above, the result of indexSequence<double>() is a batch<int64_t>,
         * and the result of indexSequence<double>() >= n is a batch_bool<int64_t>,
         * which can not be directly used as a right-hand side operand of batch_bool<double>::operator&=().
         * Defining operator&=(SimdMask const&) allows the right-hand side to be implicitly converted to SimdMask.
         *
         * @param v
         * @return SimdMask&
         */
        SimdMask& operator&=(SimdMask const& v) noexcept
        {
            XSimdType::operator&=(v);
            return *this;
        }
    };
}
