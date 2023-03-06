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

#include <blazefeo/Exception.hpp>
#include <blazefeo/Blaze.hpp>

#include <algorithm>


namespace blazefeo
{
    /**
     * @brief Interchanges two vectors
     *
     * @tparam VT0 type of first vector
     * @tparam VT1 type of second vector
     * @tparam TF transpose flag of both vectors
     *
     * @param x first vector
     * @param y second vector
     */
    template <
        typename VT0,
        typename VT1,
        bool TF
    >
    inline void swap(DenseVector<VT0, TF>& x, DenseVector<VT1, TF>& y)
    {
        auto const N = size(x);
        if (size(y) != N)
            BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Vector sizes must be equal"});

        for (size_t i = 0; i < N; ++i)
            std::swap((*x)[i], (*y)[i]);
    }
}