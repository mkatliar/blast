#pragma once

#include <blazefeo/math/dense/GemmBackend.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <algorithm>


namespace blazefeo
{
    template <typename ST1, typename MT1, typename ST2, typename MT2, typename MT3>
    void syrk_ln(
        ST1 alpha, DenseMatrix<MT1, columnMajor> const& A,
        ST2 beta, DenseMatrix<MT2, columnMajor> const& C, DenseMatrix<MT3, columnMajor>& D);


    template <typename MT1, typename MT2, typename MT3>
    BLAZE_ALWAYS_INLINE void syrk_ln(
        DenseMatrix<MT1, columnMajor> const& A,
        DenseMatrix<MT2, columnMajor> const& C, DenseMatrix<MT3, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);

        syrk_ln(ET(1.), ET(1.), ~A, ~C, ~D);
    }


    template <size_t KM, size_t KN, typename ST1, typename MT1, typename ST2, typename MT2, typename MT3>
    void syrk_ln_backend(size_t i,
        ST1 alpha, DenseMatrix<MT1, columnMajor> const& A,
        ST2 beta, DenseMatrix<MT2, columnMajor> const& C, DenseMatrix<MT3, columnMajor>& D);


    template <typename ST1, typename MT1, typename ST2, typename MT2, typename MT3>
    BLAZE_ALWAYS_INLINE void syrk_ln(
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

        size_t i = 0;

        // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
        // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
        // for (; i + 2 * TILE_SIZE < M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
        //     syrk_ln_backend<3 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, beta, ~C, ~D);

        // for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
        //     syrk_ln_backend<2 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, beta, ~C, ~D);

        for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
            syrk_ln_backend<1 * TILE_SIZE, TILE_SIZE>(i, alpha, ~A, beta, ~C, ~D);
    }


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