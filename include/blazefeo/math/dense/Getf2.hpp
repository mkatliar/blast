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
#include <blazefeo/math/dense/Swap.hpp>

#include <cmath>


namespace blazefeo
{
    /**
     * @brief computes an LU factorization of a general m-by-n matrix A
       using partial pivoting with row interchanges (unblocked algorithm).

       The factorization has the form
           A = P * L * U
       where P is a permutation matrix, L is lower triangular with unit
       diagonal elements (lower trapezoidal if m > n), and U is upper
       triangular (upper trapezoidal if m < n).

       See https://netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga8360b5b2c819e19c82bfd7e6b8285f74.html

     *
     * @tparam MT type of the matrix
     * @tparam SO storage order of the matrix
     *
     * @param A on entry, the m by n matrix to be factored.
           On exit, the factors L and U from the factorization
           A = P*L*U; the unit diagonal elements of L are not stored.
     * @param ipiv array of dimension (min(M,N))
          The pivot indices; for 0 <= i < min(M,N), row i of the
          matrix was interchanged with row ipiv[i].
     */
    template <typename MT, bool SO>
    inline void getf2(DenseMatrix<MT, SO>& A, size_t * ipiv)
    {
        using ET = ElementType_t<MT>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        for (size_t k = 0; k < M && k < N; ++k)
        {
            // Find pivot and test for singularity.
            size_t ip = k;
            auto vp = std::abs((*A)(k, k));
            for (size_t i = k + 1; i < M; ++i)
            {
                auto const v = std::abs((*A)(i, k));
                if (v > vp)
                {
                    vp = v;
                    ip = i;
                }
            }

            if (!vp)
                BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Matrix is singular"});

            // Exchange rows k and ip
            ipiv[k] = ip;
            if (ip != k)
            {
                auto x = row(*A, k);
                auto y = row(*A, ip);
                swap(x, y);
            }

            for (size_t i = k + 1; i < M; ++i)
            {
                ET const l = (*A)(i, k) / (*A)(k, k);
                (*A)(i, k) = l;

                for (size_t j = k + 1; j < N; ++j)
                    (*A)(i, j) -= l * (*A)(k, j);
            }
        }
    }
}