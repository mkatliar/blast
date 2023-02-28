// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    /**
     * @brief Computes an LU factorization of a general M-by-N matrix A
       using partial pivoting with row interchanges.

       The factorization has the form
           A = P * L * U
       where P is a permutation matrix, L is lower triangular with unit
       diagonal elements (lower trapezoidal if m > n), and U is upper
       triangular (upper trapezoidal if m < n).
     *
     * @tparam MT matrix type
     *
     * @param A on entry, the M-by-N matrix to be factored.
       On exit, the factors L and U from the factorization
       A = P*L*U; the unit diagonal elements of L are not stored.
     * @param ipiv integer array, dimension (min(M,N))
       The pivot indices; for 0 <= i < min(M,N), row i of the
       matrix was interchanged with row IPIV(i).
     */
    template <typename MT>
    inline void getrf(DenseMatrix<MT, rowMajor>& A, int * ipiv)
    {
        using ET = ElementType_t<MT>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        for (size_t k = 0; k < M && k < N; ++k)
        {
            for (size_t i = k + 1; i < M; ++i)
            {
                ET const l = (*A)(i, k) / (*A)(k, k);
                (*A)(i, k) = l;

                for (size_t j = k + 1; j < N; ++j)
                    (*A)(i, j) -= l * (*A)(k, j);
            }

            ipiv[k] = k;
        }
    }
}