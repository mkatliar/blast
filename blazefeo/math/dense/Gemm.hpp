#pragma once

#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <typename MT, bool SO>
    inline auto * ptr(DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return &(~m)(i, j);
    }


    template <typename MT, bool SO>
    inline auto const * ptr(DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return &(~m)(i, j);
    }


    template <bool SOA, bool SOB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void gemm_backend2(RegisterMatrix<T, M, N, SS>& ker, size_t K, T alpha, T beta,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd)
    {
        load2(ker, beta, c, sc);

        for (size_t k = 0; k < K; ++k)
        {
            ger2<SOA, SOB>(ker, alpha, a, sa, b, sb);

            a += SOA == columnMajor ? sa : 1;
            b += SOB == rowMajor ? sb : 1;
        }

        store2(ker, d, sd);
    }


    template <bool SOA, bool SOB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void gemm_backend2(RegisterMatrix<T, M, N, SS>& ker, size_t K, T alpha, T beta,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd,
        size_t md, size_t nd)
    {
        load2(ker, beta, c, sc, md, nd);
        
        for (size_t k = 0; k < K; ++k)
        {
            ger2<SOA, SOB>(ker, alpha, a, sa, b, sb, md, nd);

            a += SOA == columnMajor ? sa : 1;
            b += SOB == rowMajor ? sb : 1;
        }

        store2(ker, d, sd, md, nd);
    }


    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    void gemm_nt(
        ST1 alpha, ST2 beta,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, columnMajor> const& B, 
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D);


    template <typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, columnMajor> const& B, 
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        gemm_nt(ET(1.), ET(1.), ~A, ~B, ~C, ~D);
    }


    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    void gemm_nt_backend(
        size_t i, ST1 alpha, ST2 beta,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, columnMajor> const& B, 
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D);


    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        ST1 alpha, ST2 beta,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, columnMajor> const& B, 
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
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

        // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
        // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
        for (; i + 2 * TILE_SIZE < M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            gemm_nt_backend<3 * TILE_SIZE, TILE_SIZE>(i, alpha, beta, ~A, ~B, ~C, ~D);

        for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
            gemm_nt_backend<2 * TILE_SIZE, TILE_SIZE>(i, alpha, beta, ~A, ~B, ~C, ~D);

        for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
            gemm_nt_backend<1 * TILE_SIZE, TILE_SIZE>(i, alpha, beta, ~A, ~B, ~C, ~D);
    }


    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt_backend(
        size_t i, ST1 alpha, ST2 beta,
        DenseMatrix<MT1, columnMajor> const& A, DenseMatrix<MT2, columnMajor> const& B, 
        DenseMatrix<MT3, columnMajor> const& C, DenseMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_STATIC_ASSERT(KM % TILE_SIZE == 0);

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);

        BLAZE_USER_ASSERT(columns(B) == K, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(C) == M && columns(C) == N, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(D) == M && columns(D) == N, "Matrix sizes do not match");

        RegisterMatrix<ET, KM, KN, TILE_SIZE> ker;

        if (i + KM <= M)
        {
            size_t j = 0;
            ET const * a = ptr(A, i, 0);

            for (; j + KN <= N; j += KN)
                gemm_backend2<columnMajor, rowMajor>(ker, K, alpha, beta,
                    a, spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D));

            if (j < N)
                gemm_backend2<columnMajor, rowMajor>(ker, K, alpha, beta,
                    a, spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), KM, N - j);
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;
            ET const * b = ptr(B, 0, 0);

            for (; j + KN <= N; j += KN)
                gemm_backend2<columnMajor, rowMajor>(ker, K, alpha, beta,
                    ptr(A, i, 0), spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), M - i, KN);

            if (j < N)
                gemm_backend2<columnMajor, rowMajor>(ker, K, alpha, beta,
                    ptr(A, i, 0), spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), M - i, N - j);
        }
    }
}