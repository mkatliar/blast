#pragma once

#include <blazefeo/system/Tile.hpp>
#include <blazefeo/math/dense/GemmBackend.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>


namespace blazefeo
{
    template <size_t KM, size_t KN, typename ST1, typename MT1, typename ST2, typename MT2, typename MT3>
    BLAZE_ALWAYS_INLINE void syrk_ln_backend(
        size_t i, ST1 alpha,
        DenseMatrix<MT1, columnMajor> const& A,
        ST2 beta, DenseMatrix<MT2, columnMajor> const& C, DenseMatrix<MT3, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_STATIC_ASSERT(KM % TILE_SIZE == 0);

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);

        size_t const M = rows(A);
        size_t const K = columns(A);

        BLAZE_USER_ASSERT(rows(C) == M && columns(C) == M, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(D) == M && columns(D) == M, "Matrix sizes do not match");

        RegisterMatrix<ET, KM, KN, TILE_SIZE> ker;

        if (i + KM <= M)
        {
            auto a = ptr(A, i, 0);

            for (size_t j = 0; j < i; j += KN)
            {
                ker.load(beta, ptr(C, i, j));
                gemm_backend(ker, K, alpha, a, ptr(trans(A), 0, j));
                ker.store(ptr(D, i, j));
            }

            ker.load(beta, ptr(C, i, i));
            gemm_backend(ker, K, alpha, a, ptr(trans(A), 0, i));
            ker.storeLower(ptr(D, i, i));
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;
            auto b = ptr(A, 0, 0);

            for (; j < i; j += KN)
            {
                auto const md = M - i, nd = KN;
                ker.load(beta, ptr(C, i, j), md, nd);
                gemm_backend(ker, K, alpha, ptr(A, i, 0), ptr(trans(A), 0, j), md, nd);
                ker.store(ptr(D, i, j), md, nd);
            }

            if (j < M)
            {
                auto const md = M - i, nd = M - j;
                ker.load(beta, ptr(C, i, j), md, nd);
                gemm_backend(ker, K, alpha, ptr(A, i, 0), ptr(trans(A), 0, j), md, nd);
                ker.storeLower(ptr(D, i, j), md, nd);
            }
        }
    }
}