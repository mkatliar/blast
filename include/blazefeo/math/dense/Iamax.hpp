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
     * @brief Find element-wise maximum of two SIMD vectors and keep track of indices.
     *
     * This is useful for finding index of a maximum element in SIMD-enabled code.
     *
     * @tparam Float floating point vector element type
     * @tparam Index index vector element type
     *
     * @param a first floating point vector
     * @param idxa first index vector
     * @param b second floating point vector
     * @param idxb second index vector
     *
     * @return A tuple (c, idxc) where c = max(a, b), and idxc[i] = a[i] > b[i] ? idxa[i] : idxb[i]
     */
    template <typename Float, typename Index>
    requires (SimdSize_v<Float> == SimdSize_v<Index>)
    inline std::tuple<SimdVec<Float>, SimdVec<Index>> imax(
        SimdVec<Float> const& a, SimdVec<Index> const& idxa,
        SimdVec<Float> const& b, SimdVec<Index> const& idxb
    )
    {
        return std::make_tuple(max(a, b), blend(idxa, idxb, b > a));
    }


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

        size_t constexpr M = 2;

        SimdVec<ET> a[M];

        IndexType ib[M], ia[M];

        #pragma unroll
        for (size_t j = 0; j < M; ++j)
            ia[j] = ib[j] = IndexType(simd::sequenceTag, j * SS);

        size_t i = 0;

        // Load initial value
        if (M * SS <= n)
        {
            #pragma unroll
            for (size_t j = 0; j < M; ++j)
                a[j] = abs(x(j * SS).load());

            #pragma unroll
            for (size_t j = 0; j < M; ++j)
                ib[j] += M * SS;

            i += M * SS;
        }

        #pragma unroll
        for (size_t m = M; m > 0; --m)
        {
            // Process full Mx SIMD chunks
            for (; i + m * SS <= n; i += m * SS)
            {
                #pragma unroll
                for (size_t j = 0; j < m; ++j)
                    std::tie(a[j], ia[j]) = imax(a[j], ia[j], abs(x(i + j * SS).load()), ib[j]);

                #pragma unroll
                for (size_t j = 0; j < m; ++j)
                    ib[j] += m * SS;
            }

            // Reduce by 1 SIMD vector
            if (m > 1)
                std::tie(a[0], ia[0]) = imax(a[0], ia[0], a[m - 1], ia[m - 1]);
        }

        // Process the remaining elements
        if (i < n)
            std::tie(a[0], ia[0]) = imax(a[0], ia[0], abs(x(i).maskLoad(n > ib[0])), ib[0]);

        // Compute horizontal maximum
        std::tie(a[0], ia[0]) = imax(a[0], ia[0]);

        return ia[0][0];
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