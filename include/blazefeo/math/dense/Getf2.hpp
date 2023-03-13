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
#include <blazefeo/math/dense/Iamax.hpp>
#include <blazefeo/math/dense/Ger.hpp>
#include <blazefeo/math/dense/MatrixPointer.hpp>
#include <blazefeo/math/dense/VectorPointer.hpp>

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
     * @param m The number of rows of the matrix
     * @param n The number of columns of the matrix
     * @param A on entry, the m by n matrix to be factored.
           On exit, the factors L and U from the factorization
           A = P*L*U; the unit diagonal elements of L are not stored.
     * @param ipiv array of dimension (min(M,N))
          The pivot indices; for 0 <= i < min(M,N), row i of the
          matrix was interchanged with row ipiv[i].
     */
    template <typename MPA>
    requires (MatrixPointer<MPA> && StorageOrder_v<MPA> == columnMajor)
    inline void getf2(size_t m, size_t n, MPA A, size_t * ipiv)
    {
        using ET = ElementType_t<MPA>;

        for (size_t k = 0; k < m && k < n; ++k)
        {
            // Find pivot and test for singularity.
            size_t const ip = iamax(m - k, column((~A)(k, k))) + k;
            ipiv[k] = ip;

            // Exchange rows k and ip
            if (ip != k)
                swap(n, row((~A)(k, 0)), row((~A)(ip, 0)));

            if (!*(~A)(k, k))
                BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Matrix is singular"});

            for (size_t i = k + 1; i < m; ++i)
                *(~A)(i, k) /= *(~A)(k, k);

            if (k + 1 < m && k + 1 < n)
                ger(m - k - 1, n - k - 1, ET(-1), column((~A)(k + 1, k)), row((~A)(k, k + 1)), (~A)(k + 1, k + 1), (~A)(k + 1, k + 1));
        }
    }


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
    template <typename MT>
    inline void getf2(DenseMatrix<MT, columnMajor>& A, size_t * ipiv)
    {
        getf2(rows(*A), columns(*A), ptr(A), ipiv);
    }


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
    template <typename MT>
    inline void getf2(DenseMatrix<MT, rowMajor>& A, size_t * ipiv)
    {
        using ET = ElementType_t<MT>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        for (size_t k = 0; k < M && k < N; ++k)
        {
            // Find pivot and test for singularity.
            size_t const ip = iamax(subvector(column(*A, k, unchecked), k, M - k, unchecked)) + k;
            ipiv[k] = ip;

            // Exchange rows k and ip
            if (ip != k)
                swap(row(*A, k, unchecked), row(*A, ip, unchecked));

            if (!(*A)(k, k))
                BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Matrix is singular"});

            submatrix(*A, k + 1, k, M - k - 1, 1, unchecked) /= (*A)(k, k);

            ger(
                ET(-1),
                subvector(column(*A, k, unchecked), k + 1, M - k - 1, unchecked),
                subvector(row(*A, k, unchecked), k + 1, N - k - 1, unchecked),
                submatrix(*A, k + 1, k + 1, M - k - 1, N - k - 1, unchecked),
                submatrix(*A, k + 1, k + 1, M - k - 1, N - k - 1, unchecked)
            );
        }
    }
}