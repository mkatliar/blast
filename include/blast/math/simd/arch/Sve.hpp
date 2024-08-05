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
#pragma once

#include <xsimd/xsimd.hpp>

#include <type_traits>


namespace blast
{
    namespace detail
    {
        std::size_t constexpr registerCapacity(xsimd::sve)
        {
            return 32;
        }
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::sve, Arch>
    inline xsimd::batch<float, Arch> maskload(float const * src, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        throw std::logic_error {"Not implemented"};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::sve, Arch>
    inline xsimd::batch<double, Arch> maskload(double const * src, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        throw std::logic_error {"Not implemented"};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::sve, Arch>
    inline void maskstore(xsimd::batch<float, Arch> const& v, float * dst, xsimd::batch_bool<float, Arch> const& mask) noexcept
    {
        throw std::logic_error {"Not implemented"};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::sve, Arch>
    inline void maskstore(xsimd::batch<double, Arch> const& v, double * dst, xsimd::batch_bool<double, Arch> const& mask) noexcept
    {
        throw std::logic_error {"Not implemented"};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::sve, Arch>
    inline std::tuple<xsimd::batch<float, Arch>, xsimd::batch<std::int32_t, Arch>> imax(xsimd::batch<float, Arch> const& v1, xsimd::batch<std::int32_t, Arch> const& idx) noexcept
    {
        throw std::logic_error {"Not implemented"};
    }


    template <typename Arch>
    requires std::is_base_of_v<xsimd::sve, Arch>
    inline std::tuple<xsimd::batch<double, Arch>, xsimd::batch<std::int64_t, Arch>> imax(xsimd::batch<double, Arch> const& x, xsimd::batch<std::int64_t, Arch> const& idx) noexcept
    {
        throw std::logic_error {"Not implemented"};
    }
}
