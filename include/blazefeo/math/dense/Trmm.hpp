// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/dense/TrmmBackend.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <algorithm>


namespace blazefeo
{
    /// @brief C = alpha * A * B + C; A upper-triangular
    ///
    template <typename ST, typename MT1, typename MT2, bool SO2, typename MT3>
    inline void trmmLeftUpper(
        ST alpha,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, SO2> const& B,
        DenseMatrix<MT3, columnMajor>& C)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);

        size_t const M = rows(B);
        size_t const N = columns(B);

        if (rows(A) != M || columns(A) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t i = 0;

        // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
        // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
        for (; i + 2 * TILE_SIZE < M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            trmmLeftUpper_backend<3 * TILE_SIZE, TILE_SIZE>(
                M - i, N, alpha, ptr<aligned>(A, i, i), ptr<aligned>(B, i, 0), ptr<aligned>(C, i, 0));

        for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
            trmmLeftUpper_backend<2 * TILE_SIZE, TILE_SIZE>(
                M - i, N, alpha, ptr<aligned>(A, i, i), ptr<aligned>(B, i, 0), ptr<aligned>(C, i, 0));

        for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
            trmmLeftUpper_backend<1 * TILE_SIZE, TILE_SIZE>(
                M - i, N, alpha, ptr<aligned>(A, i, i), ptr<aligned>(B, i, 0), ptr<aligned>(C, i, 0));
    }


    /// @brief C = alpha * B * A + C; A lower-triangular
    ///
    template <typename ET, typename MTB, typename MTA, bool SOA, typename MTC>
    inline void trmmRightLower(
        ET alpha,
        DenseMatrix<MTB, columnMajor> const& B, DenseMatrix<MTA, SOA> const& A,
        DenseMatrix<MTC, columnMajor>& C)
    {
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MTB>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MTA>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MTC>, ET);

        size_t const M = rows(B);
        size_t const N = columns(B);

        if (rows(A) != N || columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t j = 0;

        // Main part
        for (; j + TILE_SIZE <= N; j += TILE_SIZE)
        {
            // size_t const K = N - j - TILE_SIZE;
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
				ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
				/*
                ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j));
				*/
                ker.store(ptr<aligned>(C, i, j));
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
				ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
				/*
                ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j));
				*/
                ker.store(ptr<aligned>(C, i, j));
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
				ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
				/*
                ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j));
				*/
                ker.store(ptr<aligned>(C, i, j));
            }

            // Bottom edge
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
				ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j), M - i, ker.columns());
				/*
                ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j), M - i, ker.columns());
				*/
                ker.store(ptr<aligned>(C, i, j), M - i, ker.columns());
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
                ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j), ker.rows(), N - j);
                ker.store(ptr<aligned>(C, i, j), ker.rows(), N - j);
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j), ker.rows(), N - j);
                ker.store(ptr<aligned>(C, i, j), ker.rows(), N - j);
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j), ker.rows(), N - j);
                ker.store(ptr<aligned>(C, i, j), ker.rows(), N - j);
            }

            // Bottom-right corner
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.gemm(N - j, alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j), M - i, N - j);
                ker.store(ptr<aligned>(C, i, j), M - i, N - j);
            }
        }
    }
}