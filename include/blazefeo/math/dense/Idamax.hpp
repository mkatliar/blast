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

#include <blaze/math/Aliases.h>
#include <blazefeo/Blaze.hpp>
#include <blazefeo/Exception.hpp>

#include <cmath>
#include <tuple>


namespace blazefeo
{
    /**
     * @brief Finds the index of the first element in a vector having maximum absolute value.
     *
     * https://netlib.org/lapack/explore-html/d0/d73/group__aux__blas_ga285793254ff0adaf58c605682efb880c.html
     *
     * @tparam VT vector type
     * @tparam TF vector transpose flag
     * @param x vector
     *
     * @return index of the first element in @a x having maximum absolute value.
     */
    template <typename VT, bool TF>
    inline size_t idamax(DenseVector<VT, TF> const& x)
    {
        size_t const N = size(x);
        if (N == 0)
            BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Vector is empty"});

        auto value = std::abs((*x)[0]);
        size_t index = 0;
        for (size_t i = 1; i < N; ++i)
        {
            auto const v = std::abs((*x)[i]);
            if (v > value)
            {
                value = v;
                index = i;
            }
        }

        return index;
    }
}