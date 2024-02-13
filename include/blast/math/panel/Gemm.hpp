// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/PanelMatrix.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/register_matrix/Gemm.hpp>
#include <blast/math/simd/SimdSize.hpp>
#include <blast/math/panel/MatrixPointer.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>


namespace blast
{
    template <typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor> const& B,
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        gemm_nt(ET(1.), ET(1.), *A, *B, *C, *D);
    }


    template <typename ST1, typename ST2, typename MT1, typename MT2, typename MT3, typename MT4>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        ST1 alpha, ST2 beta,
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, columnMajor> const& B,
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr SS = SimdSize_v<ET>;
        size_t constexpr TILE_STEP = 4;    // TODO: this is almost arbitrary and needs to be ppoperly determined

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
        for (; i + 2 * SS < M && i + 4 * SS != M; i += 3 * SS)
            gemm_nt_backend<3 * SS, TILE_STEP>(M, N, K, i, alpha, ptr(*A), ~trans(ptr(*B)), beta, ptr(*C), ptr(*D));

        for (; i + 1 * SS < M; i += 2 * SS)
            gemm_nt_backend<2 * SS, TILE_STEP>(M, N, K, i, alpha, ptr(*A), ~trans(ptr(*B)), beta, ptr(*C), ptr(*D));

        for (; i + 0 * SS < M; i += 1 * SS)
            gemm_nt_backend<1 * SS, TILE_STEP>(M, N, K, i, alpha, ptr(*A), ~trans(ptr(*B)), beta, ptr(*C), ptr(*D));
    }


    template <
        size_t KM, size_t KN, typename T,
        typename MPA, typename MPB, typename MPC, typename MPD
    >
    requires MatrixPointer<MPA, T> && MatrixPointer<MPB, T> && MatrixPointer<MPC, T> && MatrixPointer<MPD, T>
    BLAZE_ALWAYS_INLINE void gemm_nt_backend(size_t M, size_t N, size_t K, size_t i, T alpha, MPA A, MPB B, T beta, MPC C, MPD D)
    {
        using ET = ElementType_t<MPD>;
        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        if (i + KM <= M)
        {
            size_t j = 0;
            auto a = A(i, 0);

            for (; j + KN <= N; j += KN)
                gemm(ker, K, alpha, a, B(0, j), beta, C(i, j), D(i, j));

            if (j < N)
                gemm(ker, K, alpha, a, B(0, j), beta, C(i, j), D(i, j), KM, N - j);
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;

            for (; j + KN <= N; j += KN)
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), M - i, KN);

            if (j < N)
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), M - i, N - j);
        }
    }
}