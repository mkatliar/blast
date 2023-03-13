// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/math/StorageOrder.h>
#include <blaze/math/typetraits/StorageOrder.h>
#include <blaze/math/views/Forward.h>
#include <blazefeo/math/dense/GemmBackend.hpp>
#include <blazefeo/math/dense/MatrixPointer.hpp>

#include <algorithm>
#include <type_traits>


namespace blazefeo
{
    /**
     * @brief Performs the matrix-matrix operation
     *
     * D := alpha*A*B + beta*C
     *
     * alpha and beta are scalars, and A, B and C are matrices, with A
     * an m by k matrix, B a k by n matrix and C an m by n matrix.
     *
     * @tparam ST1
     * @tparam MPA
     * @tparam MPB
     * @tparam ST2
     * @tparam MPC
     * @tparam MPD
     *
     * @param M the number of rows of the matrices A, C, and D.
     * @param N the number of columns of the matrices B and C.
     * @param K the number of columns of the matrix A and the number of rows of the matrix B.
     * @param alpha the scalar alpha
     * @param A the matrix A
     * @param B the matrix B
     * @param beta the scalar beta
     * @param C the matrix C
     * @param D the output matrix D
     */
    template <
        typename ST1, typename MPA, typename MPB,
        typename ST2, typename MPC, typename MPD
    >
    requires (
        MatrixPointer<MPA> && StorageOrder_v<MPA> == columnMajor &&
        MatrixPointer<MPB> &&
        MatrixPointer<MPC> && StorageOrder_v<MPC> == columnMajor &&
        MatrixPointer<MPD> && StorageOrder_v<MPD> == columnMajor
    )
    inline void gemm(size_t M, size_t N, size_t K, ST1 alpha, MPA A, MPB B, ST2 beta, MPC C, MPD D)
    {
        using ET = std::remove_cv_t<ElementType_t<MPA>>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(std::remove_cv_t<ElementType_t<MPB>>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(std::remove_cv_t<ElementType_t<MPC>>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(std::remove_cv_t<ElementType_t<MPD>>, ET);

        size_t j = 0;

        // Main part
        for (; j + TILE_SIZE <= N; j += TILE_SIZE)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j));
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j));
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j));
            }

            // Bottom edge
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), M - i, ker.columns());
            }
        }


        // Right edge
        if (j < N)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), ker.rows(), N - j);
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), ker.rows(), N - j);
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), ker.rows(), N - j);
            }

            // Bottom-right corner
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), M - i, N - j);
            }
        }
    }


    /**
     * @brief Performs the matrix-matrix operation
     *
     * D := alpha*A*B + beta*C
     *
     * alpha and beta are scalars, and A, B and C are matrices, with A
     * an m by k matrix, B a k by n matrix and C an m by n matrix.
     *
     * @param M the number of rows of the matrices A, C, and D.
     * @param N the number of columns of the matrices B and C.
     * @param K the number of columns of the matrix A and the number of rows of the matrix B.
     * @param alpha the scalar alpha
     * @param A the matrix A
     * @param B the matrix B
     * @param beta the scalar beta
     * @param C the matrix C
     * @param D the output matrix D
     */
    template <
        typename ST1, typename MT1, typename MT2, bool SO2,
        typename ST2, typename MT3, typename MT4
    >
    inline void gemm(
        ST1 alpha,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, SO2> const& B,
        ST2 beta, DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        size_t const M = rows(A);
        size_t const N = columns(B);
        size_t const K = columns(A);

        if (rows(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        gemm(M, N, K, alpha, ptr(*A), ptr(*B), beta, ptr(*C), ptr(*D));
    }


    template <
        typename MT1, typename MT2, bool SO2,
        typename MT3, typename MT4
    >
    inline void gemm(
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, SO2> const& B,
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT4>;
        gemm(ET(1.), A, B, ET(1.), C, D);
    }
}