#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/panel/PanelSize.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <bool SOA, bool SOB, typename T, size_t M, size_t N>
    BLAZE_ALWAYS_INLINE void gemm_backend(RegisterMatrix<T, M, N, columnMajor>& ker, size_t K, T alpha, T beta,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd)
    {
        load(ker, beta, c, sc);

        for (size_t k = 0; k < K; ++k)
        {
            ger<SOA, SOB>(ker, alpha, a, sa, b, sb);

            a += SOA == rowMajor ? ker.panels() * sa : Simd<T>::size;
            b += SOB == rowMajor ? Simd<T>::size : N * sb;
        }

        store(ker, d, sd);
    }


    template <
        typename T, size_t M, size_t N, 
        typename MT1, bool SO1, typename MT2, bool SO2,
        typename MT3, bool SO3, typename MT4, bool SO4
        >
    BLAZE_ALWAYS_INLINE void gemm_backend(RegisterMatrix<T, M, N, columnMajor>& ker, size_t K, T alpha, T beta,
        Matrix<MT1, SO1> const& A, Matrix<MT2, SO2> const& B, Matrix<MT3, SO3> const& C, Matrix<MT4, SO4>& D)
    {
        ker.load(beta, ~C);

        for (size_t k = 0; k < K; ++k)
            ker.ger(alpha, column(~A, k), row(~B, k));

        ker.store(~D);
    }


    template <bool SOA, bool SOB, typename T, size_t M, size_t N>
    BLAZE_ALWAYS_INLINE void gemm_backend(RegisterMatrix<T, M, N, columnMajor>& ker, size_t K, T alpha, T beta,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd,
        size_t md, size_t nd)
    {
        load(ker, beta, c, sc, md, nd);
        
        for (size_t k = 0; k < K; ++k)
        {
            ger<SOA, SOB>(ker, alpha, a, sa, b, sb, md, nd);

            a += SOA == rowMajor ? ker.panels() * sa : Simd<T>::size;
            b += SOB == rowMajor ? Simd<T>::size : N * sb;
        }

        store(ker, d, sd, md, nd);
    }


    template <typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor> const& B, 
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        gemm_nt(ET(1.), ET(1.), ~A, ~B, ~C, ~D);
    }


    /// Returns the index of first unprocessed row.
    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    void gemm_nt_backend(
        size_t i, ST1 alpha, ST2 beta,
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor> const& B, 
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D);


    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        ST1 alpha, ST2 beta,
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor> const& B, 
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr PANEL_SIZE = PanelSize_v<ET>;

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

        // i + 4 * PANEL_SIZE != M is to improve performance in case when the remaining number of rows is 4 * PANEL_SIZE:
        // it is more efficient to apply 2 * PANEL_SIZE kernel 2 times than 3 * PANEL_SIZE + 1 * PANEL_SIZE kernel.
        for (; i + 2 * PANEL_SIZE < M && i + 4 * PANEL_SIZE != M; i += 3 * PANEL_SIZE)
            gemm_nt_backend<3 * PANEL_SIZE, 4>(i, alpha, beta, ~A, ~B, ~C, ~D);

        for (; i + 1 * PANEL_SIZE < M; i += 2 * PANEL_SIZE)
            gemm_nt_backend<2 * PANEL_SIZE, 4>(i, alpha, beta, ~A, ~B, ~C, ~D);

        for (; i + 0 * PANEL_SIZE < M; i += 1 * PANEL_SIZE)
            gemm_nt_backend<1 * PANEL_SIZE, 4>(i, alpha, beta, ~A, ~B, ~C, ~D);
    }


    template <size_t KM, size_t KN, typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt_backend(
        size_t i, ST1 alpha, ST2 beta,
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor> const& B, 
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr PANEL_SIZE = PanelSize_v<ET>;

        BLAZE_STATIC_ASSERT(KM % PANEL_SIZE == 0);

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);

        BLAZE_USER_ASSERT(columns(B) == K, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(C) == M && columns(C) == N, "Matrix sizes do not match");
        BLAZE_USER_ASSERT(rows(D) == M && columns(D) == N, "Matrix sizes do not match");

        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        if (i + KM <= M)
        {
            size_t j = 0;
            ET const * a = ptr(A, i, 0);

            for (; j + KN <= N; j += KN)
                gemm_backend<columnMajor, rowMajor>(ker, K, alpha, beta,
                    a, spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D));

            if (j < N)
                gemm_backend<columnMajor, rowMajor>(ker, K, alpha, beta,
                    a, spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), KM, N - j);
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;
            
            for (; j + KN <= N; j += KN)
                gemm_backend<columnMajor, rowMajor>(ker, K, alpha, beta,
                    ptr(A, i, 0), spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), M - i, KN);

            if (j < N)
                gemm_backend<columnMajor, rowMajor>(ker, K, alpha, beta,
                    ptr(A, i, 0), spacing(A), ptr(B, j, 0), spacing(B),
                    ptr(C, i, j), spacing(C), ptr(D, i, j), spacing(D), M - i, N - j);
        }
    }
}