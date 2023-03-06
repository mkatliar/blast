// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include "blazefeo/math/dense/Laswp.hpp"
#include <blaze/math/views/Forward.h>
#include <blaze/math/views/Submatrix.h>
#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/dense/Getf2.hpp>
#include <blazefeo/math/dense/Trsm.hpp>
#include <blazefeo/math/dense/Gemm.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    /**
     * @brief Computes an LU factorization of a general M-by-N matrix A
       using partial pivoting with row interchanges (column-major storage order).

       The factorization has the form
           A = P * L * U
       where P is a permutation matrix, L is lower triangular with unit
       diagonal elements (lower trapezoidal if m > n), and U is upper
       triangular (upper trapezoidal if m < n).
     *
     * The implementation is based on the Netlib implementation described here:
     * https://netlib.org/utk/papers/factor/node7.html
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
    inline void getrf(DenseMatrix<MT, columnMajor>& A, size_t * ipiv)
    {
        using ET = ElementType_t<MT>;
        size_t constexpr NB = TileSize_v<ET>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        size_t k = 0;

        for (; k + NB < M && k + NB < N; k += NB)
        {
            // Apply the LU factorization on an M x NB column panel of A (i.e., A11 and A12).
            {
                auto AA = submatrix(A, k, k, M - k, NB);
                getf2(AA, ipiv + k);
            }

            // Adjust the pivot indices.
            for (size_t i = k; i < k + NB; ++i)
                ipiv[i] += k;

            // // Apply interchanges to columns 0 ... k-1
            {
                auto AA = submatrix(A, 0, 0, M, k);
                laswp(AA, k, k + NB, ipiv);
            }

            // Apply interchanges to columns k+NB ... N-1
            {
                auto AA = submatrix(A, 0, k + NB, M, N - k - NB);
                laswp(AA, k, k + NB, ipiv);
            }

            // Compute the NB x (N - NB) row panel of U:
            // U12 := L11^{-1} A12
            auto const A11 = submatrix(A, k, k, NB, NB);
            auto A12 = submatrix(A, k, k + NB, NB, N - k - NB);
            trsm<UpLo::Lower, true>(A11, A12, A12);

            auto A22 = submatrix(A, k + NB, k + NB, M - k - NB, N - k - NB);
            gemm(
                ET(-1),
                submatrix(A, k + NB, k, M - k - NB, NB),
                submatrix(A, k, k + NB, NB, N - k - NB),
                ET(1),
                A22,
                A22
            );
        }

        {
            // Process the remaining part of the matrix with unblocked algorithm
            auto AA = submatrix(A, k, k, M - k, N - k);
            getf2(AA, ipiv + k);
        }

        // Adjust the pivot indices.
        for (size_t i = k; i < M && i < N; ++i)
            ipiv[i] += k;

        // Apply interchanges to columns 0 ... k-1
        {
            auto AA = submatrix(A, 0, 0, M, k);
            laswp(AA, k, std::min(M, N), ipiv);
        }
    }


    /**
     * @brief Computes an LU factorization of a general M-by-N matrix A
       using partial pivoting with row interchanges (row-major storage order).

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
    inline void getrf(DenseMatrix<MT, rowMajor>& A, size_t * ipiv)
    {
        using ET = ElementType_t<MT>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        for (size_t k = 0; k < M && k < N; ++k)
        {
            // subvector(column(A, k), k + 1, M - k - 1) /= (*A)(k, k);
            submatrix(A, k + 1, k, M - k - 1, 1) /= (*A)(k, k);
            // for (size_t i = k + 1; i < M; ++i)
            //     (*A)(i, k) /= (*A)(k, k);

            for (size_t j = k + 1; j < N; ++j)
                for (size_t i = k + 1; i < M; ++i)
                    (*A)(i, j) -= (*A)(i, k) * (*A)(k, j);

            ipiv[k] = k;
        }
    }
}