#pragma once

#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/system/Tile.hpp>


namespace blazefeo
{
    template <Side SIDE, UpLo UPLO, size_t KM, size_t KN, typename T, typename P1, typename P2, typename P3>
        requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T> && MatrixPointer<P3, T>
    BLAZE_ALWAYS_INLINE void trmm_backend(size_t M, size_t N, size_t K, T alpha, P1 a, P2 b, P3 c)
    {
        size_t constexpr TILE_SIZE = TileSize_v<T>;
        BLAZE_STATIC_ASSERT(KM % TILE_SIZE == 0);

        RegisterMatrix<T, KM, KN, columnMajor> ker;

        if (KM <= M)
        {
            size_t j = 0;
            
            for (; j + KN <= N; j += KN)
            {
                ker.reset();
                ker.template trmm<SIDE, UPLO>(K, alpha, a, b.offset(0, j));
                // ker.gemm(K, alpha, a, b.offset(0, j));
                ker.store(c.offset(0, j));
            }

            if (j < N)
            {
                auto const md = KM, nd = N - j;
                ker.reset();
                ker.gemm(K, alpha, a, b.offset(0, j), md, nd);
                ker.store(c.offset(0, j), md, nd);
            }
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;

            for (; j + KN <= N; j += KN)
            {
                auto const md = M, nd = KN;
                ker.reset();
                ker.gemm(K, alpha, a, b.offset(0, j), md, nd);
                ker.store(c.offset(0, j), md, nd);
            }

            if (j < N)
            {
                auto const md = M, nd = N - j;
                ker.reset();
                ker.gemm(K, alpha, a, b.offset(0, j), md, nd);
                ker.store(c.offset(0, j), md, nd);
            }
        }
    }
}