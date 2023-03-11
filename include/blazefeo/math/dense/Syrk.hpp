// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/dense/SyrkBackend.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>


namespace blazefeo
{
    template <typename ST1, typename MT1, typename ST2, typename MT2, typename MT3>
    inline void syrkLower(
        ST1 alpha,
        DenseMatrix<MT1, columnMajor> const& A,
        ST2 beta, DenseMatrix<MT2, columnMajor> const& C, DenseMatrix<MT3, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);

        size_t const M = rows(A);
        size_t const K = columns(A);

        if (rows(C) != M || columns(C) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t j = 0;

        // Main part
        for (; j + TILE_SIZE <= M; j += TILE_SIZE)
        {
            size_t i = j;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j));
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)));
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j));
                else
                    ker.store(ptr<aligned>(D, i, j));
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j));
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)));
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j));
                else
                    ker.store(ptr<aligned>(D, i, j));
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j));
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)));
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j));
                else
                    ker.store(ptr<aligned>(D, i, j));
            }

            // Bottom edge
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j), M - i, ker.columns());
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)), M - i, ker.columns());
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j), M - i, ker.columns());
                else
                    ker.store(ptr<aligned>(D, i, j), M - i, ker.columns());
            }
        }


        // Right edge
        if (j < M)
        {
            size_t i = j;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j), ker.rows(), M - j);
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)), ker.rows(), M - j);

                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j), ker.rows(), M - j);
                else
                    ker.store(ptr<aligned>(D, i, j), ker.rows(), M - j);
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j), ker.rows(), M - j);
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)), ker.rows(), M - j);
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j), ker.rows(), M - j);
                else
                    ker.store(ptr<aligned>(D, i, j), ker.rows(), M - j);
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j), ker.rows(), M - j);
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)), ker.rows(), M - j);
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j), ker.rows(), M - j);
                else
                    ker.store(ptr<aligned>(D, i, j), ker.rows(), M - j);
            }

            // Bottom-right corner
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(beta, ptr<aligned>(C, i, j), ker.rows(), M - j);
                ker.gemm(K, alpha, ptr<aligned>(A, i, 0), trans(ptr<aligned>(A, j, 0)), M - i, M - j);
                if (i == j)
                    ker.storeLower(ptr<aligned>(D, i, j), M - i, M - j);
                else
                    ker.store(ptr<aligned>(D, i, j), M - i, M - j);
            }
        }
    }
}