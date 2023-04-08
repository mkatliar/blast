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

#include <blast/math/simd/SimdSize.hpp>

#include <cstddef>


namespace blast
{
    /**
     * @brief Check if a pointer is aligned at a SIMD register boundary.
     *
     * @tparam T SIMD element type
     * @param ptr pointer
     *
     * @return true if @a ptr is a multiple of SIMD register size for type @a T
     * @return false if @a ptr is not a multiple of SIMD register size for type @a T
     */
    template <typename T>
    inline bool constexpr isSimdAligned(T * ptr) noexcept
    {
        return reinterpret_cast<std::ptrdiff_t>(ptr) % (SimdSize_v<T> * sizeof(T)) == 0;
    }
}