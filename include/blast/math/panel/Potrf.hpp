// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/PanelMatrix.hpp>
#include <blast/math/views/submatrix/Panel.hpp>
#include <blast/math/panel/Gemm.hpp>
#include <blast/math/panel/PanelSize.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blast
{
    using namespace blaze;


    template <size_t KM, size_t KN, typename MT1, typename MT2>
    BLAZE_ALWAYS_INLINE void potrf_backend(size_t k, size_t i,
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr PANEL_SIZE = PanelSize_v<ET>;

        size_t const M = rows(A);
        size_t const N = columns(A);

        BLAST_USER_ASSERT(i < M, "Index too big");
        BLAST_USER_ASSERT(k < N, "Index too big");

        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        ker.load(ptr<aligned>(*A, i, k));

        auto const a = ptr<aligned>(*L, i, 0);
        auto const b = ptr<aligned>(*L, k, 0);

        // TODO: this is a gemm(), replace by gemm()
        for (size_t l = 0; l < k; ++l)
            ker.ger(ET(-1.), column(a(0, l)), column(b(0, l)).trans());

        if (i == k)
            ker.potrf();
        else
            ker.trsm(Side::Right, UpLo::Upper, ptr<aligned>(*L, k, k).trans());

        if (k + KN <= N)
            ker.store(ptr<aligned>(*L, i, k));
        else
            ker.store(ptr<aligned>(*L, i, k), std::min(M - i, KM), N - k);
    }


    template <typename MT1, typename MT2>
    inline void potrf(
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor>& L)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr PANEL_SIZE = PanelSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);

        size_t const M = rows(A);
        size_t const N = columns(A);

        if (columns(A) > M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (rows(L) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        if (columns(L) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid matrix size");

        size_t constexpr KN = PANEL_SIZE;
        size_t k = 0;

        // This loop unroll gives some performance benefit for N >= 18,
        // but not much (about 1%).
        // #pragma unroll
        for (; k < N; k += KN)
        {
            size_t i = k;

            for (; i + 2 * PANEL_SIZE < M; i += 3 * PANEL_SIZE)
                potrf_backend<3 * PANEL_SIZE, KN>(k, i, *A, *L);

            for (; i + 1 * PANEL_SIZE < M; i += 2 * PANEL_SIZE)
                potrf_backend<2 * PANEL_SIZE, KN>(k, i, *A, *L);

            for (; i + 0 * PANEL_SIZE < M; i += 1 * PANEL_SIZE)
                potrf_backend<1 * PANEL_SIZE, KN>(k, i, *A, *L);
        }
    }
}
