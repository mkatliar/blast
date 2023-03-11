// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/dense/GemmBackend.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <algorithm>


namespace blazefeo
{
    template <
        typename ST1, typename MT1, typename MT2, bool SO2,
        typename ST2, typename MT3, typename MT4
    >
    inline void gemm(
        ST1 alpha,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, SO2> const& B,
        ST2 beta, DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = columns(B);
        size_t const K = columns(A);

        bool constexpr A_aligned = IsAligned_v<MT1>;
        bool constexpr B_aligned = IsAligned_v<MT2>;
        bool constexpr C_aligned = IsAligned_v<MT3>;
        bool constexpr D_aligned = IsAligned_v<MT4>;

        if (rows(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

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
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j));
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j));
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j));
            }

            // Bottom edge
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), M - i, ker.columns());
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
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), ker.rows(), N - j);
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), ker.rows(), N - j);
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), ker.rows(), N - j);
            }

            // Bottom-right corner
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, alpha, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), beta, ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), M - i, N - j);
            }
        }
    }


    template <
        typename MT1, typename MT2, bool SO2,
        typename MT3, typename MT4
    >
    inline void gemm(
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, SO2> const& B,
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = columns(B);
        size_t const K = columns(A);

        bool constexpr A_aligned = IsAligned_v<MT1>;
        bool constexpr B_aligned = IsAligned_v<MT2>;
        bool constexpr C_aligned = IsAligned_v<MT3>;
        bool constexpr D_aligned = IsAligned_v<MT4>;

        if (rows(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

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
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j));
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j));
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j));
            }

            // Bottom edge
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), M - i, ker.columns());
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
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), ker.rows(), N - j);
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), ker.rows(), N - j);
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), ker.rows(), N - j);
            }

            // Bottom-right corner
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                gemm(ker, K, ptr<A_aligned>(*A, i, 0), ptr<B_aligned>(*B, 0, j), ptr<C_aligned>(*C, i, j), ptr<D_aligned>(*D, i, j), M - i, N - j);
            }
        }
    }
}