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
#include <blazefeo/math/dense/Swap.hpp>


namespace blazefeo
{
    /**
     * @brief Performs a series of row interchanges on a general rectangular matrix.
     * See https://netlib.org/lapack/explore-html/d8/d9b/group__double_o_t_h_e_rauxiliary_ga3ccc0cf84b0493bd9adcdc02fcff449f.html
     *
     * @tparam MT matrix type
     * @tparam SO matrix storage order
     *
     * @param A on entry, the matrix to which the row interchanges will be applied. On exit, the permuted matrix.
     * @param k0 The first element of @a ipiv for which a row interchange will be done.
     * @param k1 @a k1 - @a k0 is the number of elements of @a ipiv for which a row interchange will be done.
     * @param ipiv The vector of pivot indices of size @a k1. Only the elements in positions
          @a k0 ... @a k1 - 1 of @a ipiv are accessed. ipiv[k] = l implies rows k and l are to be interchanged.
     */
    template <typename MT, bool SO>
    inline void laswp(DenseMatrix<MT, SO>& A, size_t k0, size_t k1, size_t * ipiv)
    {
        for (size_t k = k0; k < k1; ++k)
        {
            auto r0 = row(*A, k);
            auto r1 = row(*A, ipiv[k]);
            swap(r0, r1);
        }
    }
}