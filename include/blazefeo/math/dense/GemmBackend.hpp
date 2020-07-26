#pragma once

#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/system/Tile.hpp>


namespace blazefeo
{
    template <
        typename T, size_t M, size_t N, size_t SS,
        typename PA, typename PB
    >
        requires MatrixPointer<PA, columnMajor> && MatrixPointer<PB, rowMajor>
    BLAZE_ALWAYS_INLINE void gemm_backend(
        RegisterMatrix<T, M, N, SS>& ker, size_t K, T alpha, PA a, PB b)
    {
        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(alpha, a, b);
            a.hmove(1);
            b.vmove(1);
        }
    }


    template <
        typename T, size_t M, size_t N, size_t SS,
        typename PA, typename PB
    >
        requires MatrixPointer<PA, columnMajor> && MatrixPointer<PB, rowMajor>
    BLAZE_ALWAYS_INLINE void gemm_backend(RegisterMatrix<T, M, N, SS>& ker, size_t K,
        T alpha, PA a, PB b, size_t md, size_t nd)
    {
        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(alpha, a, b, md, nd);
            a.hmove(1);
            b.vmove(1);
        }
    }


    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_backend(
        size_t i,
        ST1 alpha, DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, rowMajor> const& B, 
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

        BLAZE_USER_ASSERT(rows(B) == K, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(C) == M && columns(C) == N, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(D) == M && columns(D) == N, "Matrix sizes do not match");

        RegisterMatrix<ET, KM, KN, TILE_SIZE> ker;

        if (i + KM <= M)
        {
            size_t j = 0;
            auto a = ptr(A, i, 0);

            for (; j + KN <= N; j += KN)
            {
                ker.load(beta, ptr(C, i, j));
                gemm_backend(ker, K, alpha, a, ptr(B, 0, j));
                ker.store(ptr(D, i, j));
            }

            if (j < N)
            {
                auto const md = KM, nd = N - j;
                ker.load(beta, ptr(C, i, j), md, nd);
                gemm_backend(ker, K, alpha, a, ptr(B, 0, j), md, nd);
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
                gemm_backend(ker, K, alpha, ptr(A, i, 0), ptr(B, 0, j), md, nd);
                ker.store(ptr(D, i, j), md, nd);
            }

            if (j < N)
            {
                auto const md = M - i, nd = N - j;
                ker.load(beta, ptr(C, i, j), md, nd);
                gemm_backend(ker, K, alpha, ptr(A, i, 0), ptr(B, 0, j), md, nd);
                ker.store(ptr(D, i, j), md, nd);
            }
        }
    }
}