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
    template <typename T, typename Arch>
    using SimdMask = xsimd::batch_bool<T, Arch>;
}
