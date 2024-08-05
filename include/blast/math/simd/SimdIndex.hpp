// Copyright 2024 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma  once

#include <blast/math/simd/Simd.hpp>

#include <cstdint>
#include <type_traits>


namespace blast
{
    namespace detail
    {
        /// @brief Deduces integer type of given size, sized or unsigned.
        ///
        /// @tparam N type size in bytes
        /// @tparam S true for signed type, false for unsigned
        ///
        template <std::size_t N, bool S>
        struct IntOfSize;

        template <std::size_t N, bool S>
        using IntOfSize_t = typename IntOfSize<N, S>::Type;

        template <>
        struct IntOfSize<4, true>
        {
            using Type = std::int32_t;
        };

        template <>
        struct IntOfSize<4, false>
        {
            using Type = std::uint32_t;
        };

        template <>
        struct IntOfSize<8, true>
        {
            using Type = std::int64_t;
        };

        template <>
        struct IntOfSize<8, false>
        {
            using Type = std::uint64_t;
        };

        template <typename T, typename Arch>
        requires (xsimd::batch<T, Arch>::size == 2) && std::is_integral_v<T>
        constexpr xsimd::batch<T, Arch> integerSequence()
        {
            return {0, 1};
        }

        template <typename T, typename Arch>
        requires (xsimd::batch<T, Arch>::size == 4) && std::is_integral_v<T>
        constexpr xsimd::batch<T, Arch> integerSequence()
        {
            return {0, 1, 2, 3};
        }

        template <typename T, typename Arch = xsimd::default_arch>
        requires (xsimd::batch<T, Arch>::size == 8) && std::is_integral_v<T>
        constexpr xsimd::batch<T, Arch> integerSequence()
        {
            return {0, 1, 2, 3, 4, 5, 6, 7};
        }
    }


    /// @brief Integer SIMD type that can be used for indexing or gather-scatter operations.
    ///
    /// @tparam T the deduced type will be the same size as @a T
    /// @tparam Arch instruction set architecture
    ///
    template <typename T, typename Arch = xsimd::default_arch>
    using SimdIndex = xsimd::batch<detail::IntOfSize_t<sizeof(T), true>, Arch>;


    /// @brief Construct an integer index sequence
    ///
    /// @return [0, 1, ..., SimdIndex<T, Arch>::size - 1]
    ///
    template <typename T, typename Arch = xsimd::default_arch>
    constexpr SimdIndex<T, Arch> indexSequence()
    {
        return detail::integerSequence<detail::IntOfSize_t<sizeof(T), true>, Arch>();
    }
}
