// Copyright 2023-2024 Mikhail Katliar
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

#include <cstdlib>
#include <type_traits>


namespace blast
{
    /**
     * @brief Size of a SIMD register conatining scalars of a given type.
     *
     * @tparam T scalar type
     * @tparam Arch instruction set architecture
     */
    template <typename T, typename Arch = xsimd::default_arch>
    std::size_t constexpr SimdSize_v = xsimd::batch<std::remove_cv_t<T>, Arch>::size;
}
