#pragma once

#include <smoke/StaticMatrix.hpp>
#include <smoke/GemmKernel.hpp>
#include <smoke/GemmKernel_double_3_1_4.hpp>

#include <algorithm>

namespace smoke
{
    template <typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_nt(
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, N, K, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        size_t i = 0;

        for (; (i + 3) * P <= M; i += 3)
        {
            GemmKernel<T, 3, 1, P, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * P <= N; j += 1)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing());

            for (; j * P < N; ++j)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), 3 * P, std::min(N - j * P, P));
        }

        for (; (i + 2) * P <= M; i += 2)
        {
            GemmKernel<T, 2, 1, P, false, true> ker;
            size_t j = 0;

            for (; (j + 1) * P <= N; j += 1)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing());

            for (; j * P < N; ++j)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), 2 * P, std::min(N - j * P, P));
        }

        for (; i * P < M; ++i)
        {
            GemmKernel<T, 1, 1, P, false, true> ker;

            size_t const rm = std::min(M - i * P, P);
            size_t j = 0;

            for (; (j + 1) * P <= N; j += 1)
                gemm(ker, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), rm, 1 * P);

            for (; j * P < N; ++j)
                gemm(ker, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), rm, std::min(N - j * P, P));
        }
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm(GemmKernel<T, KM, KN, P, false, true> ker,
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, N, K, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        // size_t constexpr MR = M % (KM * P);
        // size_t constexpr NR = N % (KN * P);
        // size_t constexpr MM = M / P;
        // size_t constexpr NN = N / P;
        // size_t constexpr MM_MAX = MM + (M % P > 0);
        // size_t constexpr NN_MAX = NN + (N % P > 0);

        size_t i = 0;

        for (; (i + KM) * P <= M; i += KM)
        {
            size_t j = 0;

            for (; (j + KN) * P <= N; j += KN)
                gemm(ker, K,
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing());

            for (; j * P < N; ++j)
                gemm(GemmKernel<T, KM, 1, P, false, true> {}, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), KM * P, std::min(N - j * P, P));
        }

        for (; i * P < M; ++i)
        {
            size_t const rm = std::min(M - i * P, P);
            size_t j = 0;

            for (; (j + KN) * P <= N; j += KN)
                gemm(GemmKernel<T, 1, KN, P, false, true> {}, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), rm, KN * P);

            for (; j * P < N; ++j)
                gemm(GemmKernel<T, 1, 1, P, false, true> {}, K, 
                    A.block(i, 0), A.spacing(), B.block(j, 0), B.spacing(),
                    C.block(i, j), C.spacing(), D.block(i, j), D.spacing(), rm, std::min(N - j * P, P));
        }
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm(GemmKernel<T, KM, KN, P, true, false> ker,
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        
        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                gemm(ker, K, 
                    A.block(0, KM * i), A.spacing(), B.block(0, KN * j), B.spacing(),
                    C.block(KM * i, KN * j), C.spacing(), D.block(KM * i, KN * j), D.spacing());
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm(GemmKernel<T, KM, KN, P, false, false> ker,
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                gemm(ker, K, 
                    A.block(KM * i, 0), A.spacing(), B.block(0, KN * j), B.spacing(),
                    C.block(KM * i, KN * j), C.spacing(), D.block(KM * i, KN * j), D.spacing());
    }
}