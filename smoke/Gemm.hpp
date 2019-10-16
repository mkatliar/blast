#pragma once

#include <smoke/StaticMatrix.hpp>
#include <smoke/GemmKernel.hpp>


namespace smoke
{
    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm(GemmKernel<T, KM, KN, P, false, true>,
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, N, K, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                gemm(GemmKernel<T, KM, KN, P, false, true> {}, K, 
                    A.block(KM * i, 0), A.spacing(), B.block(KN * j, 0), B.spacing(),
                    C.block(KM * i, KN * j), C.spacing(), D.block(KM * i, KN * j), D.spacing());
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm(GemmKernel<T, KM, KN, P, true, false>,
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                gemm(GemmKernel<T, KM, KN, P, true, false> {}, K, 
                    A.block(0, KM * i), A.spacing(), B.block(0, KN * j), B.spacing(),
                    C.block(KM * i, KN * j), C.spacing(), D.block(KM * i, KN * j), D.spacing());
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm(GemmKernel<T, KM, KN, P, false, false>,
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                gemm(GemmKernel<T, KM, KN, P, false, false> {}, K, 
                    A.block(KM * i, 0), A.spacing(), B.block(0, KN * j), B.spacing(),
                    C.block(KM * i, KN * j), C.spacing(), D.block(KM * i, KN * j), D.spacing());
    }
}