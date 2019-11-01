#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_1_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_2_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_3_1_4.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        PanelMatrix<MT1, rowMajor> const& A, PanelMatrix<MT2, rowMajor> const& B, 
        PanelMatrix<MT3, rowMajor> const& C, PanelMatrix<MT4, rowMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);

        if (columns(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t i = 0;
        ET const * a = tile(A, 0, 0);

        for (; (i + 3) * TILE_SIZE <= M; i += 3, a += 3 * spacing(A))
        {
            GemmKernel<ET, 3, 1, TILE_SIZE, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D));

            for (; j * TILE_SIZE < N; ++j)
                gemm(ker, K,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), 3 * TILE_SIZE, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }

        for (; (i + 2) * TILE_SIZE <= M; i += 2, a += 2 * spacing(A))
        {
            GemmKernel<ET, 2, 1, TILE_SIZE, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D));

            for (; j * TILE_SIZE < N; ++j)
                gemm(ker, K,
                    a, spacing(A), tile(B, j, 0), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), 2 * TILE_SIZE, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }

        for (; (i + 1) * TILE_SIZE <= M; ++i, a += spacing(A))
        {
            GemmKernel<ET, 1, 1, TILE_SIZE, false, true> ker;

            size_t j = 0;
            ET const * b = tile(B, 0, 0);

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D));

            for (; j * TILE_SIZE < N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), TILE_SIZE, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }

        for (; i * TILE_SIZE < M; ++i, a += spacing(A))
        {
            GemmKernel<ET, 1, 1, TILE_SIZE, false, true> ker;

            size_t const rm = std::min(M - i * TILE_SIZE, TILE_SIZE);
            size_t j = 0;
            ET const * b = tile(B, 0, 0);

            for (; (j + 1) * TILE_SIZE <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), rm, TILE_SIZE);

            for (; j * TILE_SIZE < N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    tile(C, i, j), spacing(C), tile(D, i, j), spacing(D), rm, std::min(N - j * TILE_SIZE, TILE_SIZE));
        }
    }
}