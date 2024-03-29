// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/simd/RegisterMatrix.hpp>
#include <blast/math/dense/DynamicMatrixPointer.hpp>
#include <blast/math/dense/StaticMatrixPointer.hpp>
#include <blast/system/Tile.hpp>


namespace blast
{
    template <
        size_t KM, size_t KN,
        typename ST1, typename MT1, typename MT2, bool SO2,
        typename ST2, typename MT3, typename MT4
    >
    BLAZE_ALWAYS_INLINE void gemm_backend(
        size_t i,
        ST1 alpha, DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, SO2> const& B,
        ST2 beta, DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_STATIC_ASSERT(KM % TILE_SIZE == 0);

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = columns(B);
        size_t const K = columns(A);

        BLAST_USER_ASSERT(rows(B) == K, "Matrix sizes do not match");
        BLAST_USER_ASSERT(rows(C) == M && columns(C) == N, "Matrix sizes do not match");
        BLAST_USER_ASSERT(rows(D) == M && columns(D) == N, "Matrix sizes do not match");

        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        if (i + KM <= M)
        {
            size_t j = 0;
            auto a = ptr(A, i, 0);

            for (; j + KN <= N; j += KN)
            {
                ker.load(beta, ptr(C, i, j));
                ker.gemm(K, alpha, a, ptr(B, 0, j));
                ker.store(ptr(D, i, j));
            }

            if (j < N)
            {
                auto const md = KM, nd = N - j;
                ker.load(beta, ptr(C, i, j), md, nd);
                ker.gemm(K, alpha, a, ptr(B, 0, j), md, nd);
                ker.store(ptr(D, i, j), md, nd);
            }
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;
            auto b = ptr(B, 0, 0);

            for (; j + KN <= N; j += KN)
            {
                auto const md = M - i, nd = KN;
                ker.load(beta, ptr(C, i, j), md, nd);
                ker.gemm(K, alpha, ptr(A, i, 0), ptr(B, 0, j), md, nd);
                ker.store(ptr(D, i, j), md, nd);
            }

            if (j < N)
            {
                auto const md = M - i, nd = N - j;
                ker.load(beta, ptr(C, i, j), md, nd);
                ker.gemm(K, alpha, ptr(A, i, 0), ptr(B, 0, j), md, nd);
                ker.store(ptr(D, i, j), md, nd);
            }
        }
    }
}