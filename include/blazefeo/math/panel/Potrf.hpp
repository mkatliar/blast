#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>
#include <blazefeo/math/panel/Gemm.hpp>
#include <blazefeo/math/panel/PanelSize.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
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

        BLAZE_USER_ASSERT(i < M, "Index too big");
        BLAZE_USER_ASSERT(k < N, "Index too big");

        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        load(ker, ptr(A, i, k), spacing(A));

        ET const * a = ptr(L, i, 0);
        ET const * b = ptr(L, k, 0);

        for (size_t l = 0; l < k; ++l)
            ger<MT1::storageOrder, !MT2::storageOrder>(ker, ET(-1.), a + PANEL_SIZE * l, spacing(L), b + PANEL_SIZE * l, spacing(L));

        if (i == k)
            ker.potrf();
        else
            trsm<false, false, true>(ker, ptr(L, k, k), spacing(L));

        if (k + KN <= N)
            store(ker, ptr(L, i, k), spacing(L));
        else
            store(ker, ptr(L, i, k), spacing(L), std::min(M - i, KM), N - k);
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

        size_t constexpr KN = 4;
        size_t k = 0;

        // This loop unroll gives some performance benefit for N >= 18,
        // but not much (about 1%).
        // #pragma unroll
        for (; k < N; k += KN)
        {
            size_t i = k;

            for (; i + 2 * PANEL_SIZE < M; i += 3 * PANEL_SIZE)
                potrf_backend<3 * PANEL_SIZE, KN>(k, i, ~A, ~L);

            for (; i + 1 * PANEL_SIZE < M; i += 2 * PANEL_SIZE)
                potrf_backend<2 * PANEL_SIZE, KN>(k, i, ~A, ~L);

            for (; i + 0 * PANEL_SIZE < M; i += 1 * PANEL_SIZE)
                potrf_backend<1 * PANEL_SIZE, KN>(k, i, ~A, ~L);
        }
    }
}