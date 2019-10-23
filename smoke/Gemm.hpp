#pragma once

#include <smoke/StaticPanelMatrix.hpp>
#include <smoke/gemm/GemmKernel.hpp>
#include <smoke/gemm/GemmKernel_double_1_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_2_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_3_1_4.hpp>

#include <algorithm>

namespace smoke
{
    template <size_t M, size_t N, size_t K>
    inline void gemm_nt(
        StaticPanelMatrix<double, M, K, 4> const& A, StaticPanelMatrix<double, N, K, 4> const& B, 
        StaticPanelMatrix<double, M, N, 4> const& C, StaticPanelMatrix<double, M, N, 4>& D)
    {
        size_t i = 0;

        for (; (i + 3) * 4 <= M; i += 3)
        {
            GemmKernel<double, 3, 1, 4, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * 4 <= N; j += 1)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing());

            for (; j * 4 < N; ++j)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), 12, std::min(N - j * 4ul, 4ul));
        }

        for (; (i + 2) * 4 <= M; i += 2)
        {
            GemmKernel<double, 2, 1, 4, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * 4 <= N; j += 1)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing());

            for (; j * 4 < N; ++j)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), 8, std::min(N - j * 4ul, 4ul));
        }

        for (; i * 4 < M; ++i)
        {
            GemmKernel<double, 1, 1, 4, false, true> ker;

            size_t const rm = std::min(M - i * 4ul, 4ul);
            size_t j = 0;

            for (; (j + 1) * 4 <= N; j += 1)
                gemm(ker, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), rm, 4);

            for (; j * 4 < N; ++j)
                gemm(ker, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), rm, std::min(N - j * 4ul, 4ul));
        }
    }
}