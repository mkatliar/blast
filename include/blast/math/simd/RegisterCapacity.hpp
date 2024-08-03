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

#include <blast/math/simd/Simd.hpp>

#include <cstdlib>


namespace blast
{
    /**
     * @brief Number of available SIMD registers.
     *
     * @return Number of SIMD registers for AVX2
     */
    std::size_t constexpr registerCapacity(xsimd::avx2)
    {
        return 16;
    }
}
