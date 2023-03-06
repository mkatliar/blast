// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

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
       using partial pivoting with row interchanges.

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
    inline void getrf(DenseMatrix<MT, columnMajor>& A, int * ipiv)
    {
        using ET = ElementType_t<MT>;
        size_t constexpr NB = TileSize_v<ET>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        size_t k = 0;

        for (; k < M && k < N; k += NB)
        {
            size_t const jb = std::min(std::min(M, N) - k, NB);

            // Apply the LU factorization on an M x NB column panel of A (i.e., A11 and A12).
            {
                auto A11A12 = submatrix(A, k, k, M - k, jb);
                getf2(A11A12, ipiv + k);
            }

            // Compute the NB x (N - NB) row panel of U:
            // U12 := L11^{-1} A12
            if (k + jb < N)
            {
                auto const A11 = submatrix(A, k, k, jb, jb);
                auto A12 = submatrix(A, k, k + jb, jb, N - k - jb);
                trsm<UpLo::Lower, true>(A11, A12, A12);

                if (k + jb < M)
                {
                    auto A22 = submatrix(A, k + jb, k + jb, M - k - jb, N - k - jb);
                    gemm(
                        ET(-1),
                        submatrix(A, k + jb, k, M - k - jb, jb),
                        submatrix(A, k, k + jb, jb, N - k - jb),
                        ET(1),
                        A22,
                        A22
                    );
                }
            }
        }

        // for (size_t k = 0; k < M && k < N; ++k)
        // {
        //     for (size_t i = k + 1; i < M; ++i)
        //     {
        //         ET const l = (*A)(i, k) / (*A)(k, k);
        //         (*A)(i, k) = l;

        //         for (size_t j = k + 1; j < N; ++j)
        //             (*A)(i, j) -= l * (*A)(k, j);
        //     }

        //     ipiv[k] = k;
        // }
    }
}