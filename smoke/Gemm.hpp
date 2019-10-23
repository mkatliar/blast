#pragma once

#include <smoke/PanelMatrix.hpp>
#include <smoke/gemm/GemmKernel.hpp>
#include <smoke/gemm/GemmKernel_double_1_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_2_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_3_1_4.hpp>

#include <blaze/util/Exception.h>

#include <algorithm>


namespace smoke
{
    template <typename MT1, typename MT2, typename MT3, typename MT4>
    inline void gemm_nt(
        PanelMatrix<MT1, 4> const& A, PanelMatrix<MT2, 4> const& B, 
        PanelMatrix<MT3, 4> const& C, PanelMatrix<MT4, 4>& D)
    {
        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);

        if (columns(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t i = 0;

        for (; (i + 3) * 4 <= M; i += 3)
        {
            GemmKernel<double, 3, 1, 4, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * 4 <= N; j += 1)
                gemm(ker, K,
                    block(A, i, 0), spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D));

            for (; j * 4 < N; ++j)
                gemm(ker, K,
                    block(A, i, 0), spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), 12, std::min(N - j * 4ul, 4ul));
        }

        for (; (i + 2) * 4 <= M; i += 2)
        {
            GemmKernel<double, 2, 1, 4, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * 4 <= N; j += 1)
                gemm(ker, K,
                    block(A, i, 0), spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D));

            for (; j * 4 < N; ++j)
                gemm(ker, K,
                    block(A, i, 0), spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), 8, std::min(N - j * 4ul, 4ul));
        }

        for (; i * 4 < M; ++i)
        {
            GemmKernel<double, 1, 1, 4, false, true> ker;

            size_t const rm = std::min(M - i * 4ul, 4ul);
            size_t j = 0;

            for (; (j + 1) * 4 <= N; j += 1)
                gemm(ker, K, 
                    block(A, i, 0), spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), rm, 4);

            for (; j * 4 < N; ++j)
                gemm(ker, K, 
                    block(A, i, 0), spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), rm, std::min(N - j * 4ul, 4ul));
        }
    }
}