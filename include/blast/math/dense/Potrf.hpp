// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


#include <blast/math/Matrix.hpp>
#include <blast/math/RowColumnVectorPointer.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/system/Tile.hpp>

#include <blast/blaze/Math.hpp>

#include <algorithm>


namespace blast
{
    template <size_t KM, size_t KN, typename MT1, typename MT2>
    BLAZE_ALWAYS_INLINE void potrf_backend(size_t k, size_t i,
        blaze::DenseMatrix<MT1, columnMajor> const& A, blaze::DenseMatrix<MT2, columnMajor>& L)
    {
        using ET = blaze::ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        BLAST_USER_ASSERT(i < M, "Index too big");
        BLAST_USER_ASSERT(k < N, "Index too big");

        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        ker.load(1., ptr<aligned>(*A, i, k));

        auto a = ptr<aligned>(*L, i, 0);
        auto b = ptr<aligned>(*L, k, 0);

        for (size_t l = 0; l < k; ++l)
            ker.ger(ET(-1.), column(a(0, l)), row(trans(b)(l, 0)));

        if (i == k)
        {
            // Diagonal blocks
            ker.potrf();

            if (k + KN <= N)
                ker.storeLower(ptr<aligned>(*L, i, k));
            else
                ker.storeLower(ptr<aligned>(*L, i, k), std::min(M - i, KM), N - k);
        }
        else
        {
            // Off-diagonal blocks
            ker.trsm(Side::Right, UpLo::Upper, ptr<aligned>(*L, k, k).trans());

            if (k + KN <= N)
                ker.store(ptr<aligned>(*L, i, k));
            else
                ker.store(ptr<aligned>(*L, i, k), std::min(M - i, KM), N - k);
        }
    }


    template <typename MT1, typename MT2>
    inline void potrf(
        blaze::DenseMatrix<MT1, columnMajor> const& A, blaze::DenseMatrix<MT2, columnMajor>& L)
    {
        using ET = blaze::ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(blaze::ElementType_t<MT2>, ET);

        size_t const M = rows(A);
        size_t const N = columns(A);

        if (columns(A) > M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (rows(L) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (columns(L) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        size_t constexpr KN = 4;
        size_t k = 0;

        // This loop unroll gives some performance benefit for N >= 18,
        // but not much (about 1%).
        // #pragma unroll
        for (; k < N; k += KN)
        {
            size_t i = k;

            for (; i + 2 * TILE_SIZE < M; i += 3 * TILE_SIZE)
                potrf_backend<3 * TILE_SIZE, KN>(k, i, *A, *L);

            for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
                potrf_backend<2 * TILE_SIZE, KN>(k, i, *A, *L);

            for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
                potrf_backend<1 * TILE_SIZE, KN>(k, i, *A, *L);
        }
    }
}
