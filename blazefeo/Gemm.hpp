#pragma once

#include <blazefeo/PanelMatrix.hpp>
#include <blazefeo/gemm/GemmKernel.hpp>
#include <blazefeo/gemm/GemmKernel_double_1_1_4.hpp>
#include <blazefeo/gemm/GemmKernel_double_2_1_4.hpp>
#include <blazefeo/gemm/GemmKernel_double_3_1_4.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <typename MT1, typename MT2, typename MT3, typename MT4, size_t P>
    BLAZE_ALWAYS_INLINE void gemm_nt(
        PanelMatrix<MT1, P> const& A, PanelMatrix<MT2, P> const& B, 
        PanelMatrix<MT3, P> const& C, PanelMatrix<MT4, P>& D)
    {
        using ET = ElementType_t<MT1>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT2>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT3>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT4>, ET);

        size_t const M = rows(A);
        size_t const N = rows(B);
        size_t const K = columns(A);
        size_t constexpr block_element_count = P * P;

        if (columns(B) != K)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(D) != M || columns(D) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        size_t const j1 = N / P;
        size_t const rn = N % P;

        size_t i = 0;
        ET const * a = block(A, 0, 0);

        for (; (i + 3) * P <= M; i += 3, a += 3 * spacing(A))
        {
            GemmKernel<ET, 3, 1, P, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * P <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D));

            for (; j * P < N; ++j)
                gemm(ker, K,
                    a, spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), 3 * P, std::min(N - j * P, P));
        }

        for (; (i + 2) * P <= M; i += 2, a += 2 * spacing(A))
        {
            GemmKernel<ET, 2, 1, P, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * P <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D));

            for (; j * P < N; ++j)
                gemm(ker, K,
                    a, spacing(A), block(B, j, 0), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), 2 * P, std::min(N - j * P, P));
        }

        for (; (i + 1) * P <= M; ++i, a += spacing(A))
        {
            GemmKernel<ET, 1, 1, P, false, true> ker;

            size_t j = 0;
            ET const * b = block(B, 0, 0);

            for (; (j + 1) * P <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D));

            for (; j * P < N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), P, std::min(N - j * P, P));
        }

        for (; i * P < M; ++i, a += spacing(A))
        {
            GemmKernel<ET, 1, 1, P, false, true> ker;

            size_t const rm = std::min(M - i * P, P);
            size_t j = 0;
            ET const * b = block(B, 0, 0);

            for (; (j + 1) * P <= N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), rm, P);

            for (; j * P < N; ++j)
                gemm(ker, K,
                    a, spacing(A), b + j * spacing(B), spacing(B),
                    block(C, i, j), spacing(C), block(D, i, j), spacing(D), rm, std::min(N - j * P, P));
        }
    }
}