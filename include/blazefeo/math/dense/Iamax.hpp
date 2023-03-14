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

#include <blazefeo/Blaze.hpp>
#include <blazefeo/Exception.hpp>
#include <blazefeo/math/dense/VectorPointer.hpp>
#include <blazefeo/math/simd/Avx256.hpp>

#include <cmath>
#include <tuple>


namespace blazefeo
{
    /**
     * @brief Finds the index of the first element in a vector having maximum absolute value.
     *
     * https://netlib.org/lapack/explore-html/d0/d73/group__aux__blas_ga285793254ff0adaf58c605682efb880c.html
     *
     * @tparam TF vector orientation
     * @tparam MP matrix pointer type
     *
     * @param n size of the vector
     * @param x matrix pointer to the vector
     *
     * @return index of the first element in @a x having maximum absolute value.
     */
    template <typename VP>
    requires VectorPointer<VP>
    inline size_t iamax(size_t n, VP x)
    {
        BLAZE_USER_ASSERT(n > 0, "Vector must be non-empty");

        using ET = std::remove_cv_t<ElementType_t<VP>>;
        size_t constexpr SS = SimdVec<ET>::size();
        using IndexType = IntVecType_t<SS>;

        SimdVec<ET> a;
        IndexType ib {simd::sequenceTag, 0};
        IndexType ia;

        ET value;
        size_t index;
        size_t i = 0;

        if (i + 2 * SS <= n)
        {
            a = abs(x(i).load());
            ia = ib;
            i += SS;
            ib += SS;

            for (; i + SS <= n; i += SS, ib += SS)
            {
                SimdVec<ET> const b = abs(x(i).load());
                auto const mask = b > a;
                a = blend(a, b, mask);
                ia = blend(ia, ib, mask);
            }

            value = std::abs(a[0]);
            index = ia[0];
            for (int j = 1; j < SS; ++j)
                if (a[j] > value)
                {
                    value = a[j];
                    index = ia[j];
                }
        }
        else
        {
            value = std::abs(*x);
            index = 0;
            ++i;
        }

        for (; i < n; ++i)
        {
            ET const v = std::abs(*(~x)(i));
            if (v > value)
            {
                value = v;
                index = i;
            }
        }

        return index;
    }


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
    inline size_t iamax(DenseVector<VT, TF> const& x)
    {
        size_t const N = size(x);
        if (N == 0)
            BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Vector is empty"});

        return iamax(N, ptr(*x));
    }
}