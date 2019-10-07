#pragma once

#include <smoke/StaticMatrix.hpp>
#include <smoke/GemmKernel.hpp>


namespace smoke
{
    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_nt(
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
            {
                GemmKernel<T, KM, KN, P> ker;
                ker.load(C.block(KM * i, KN * j), C.spacing());

                for (size_t k = 0; k < KK; ++k)
                    ker(A.block(KM * i, k), A.spacing(), false, B.block(KN * j, k), B.spacing(), true);

                ker.store(D.block(KM * i, KN * j), D.spacing());
            }
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_tn(
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
            {
                GemmKernel<T, KM, KN, P> ker;
                ker.load(C.block(KM * i, KN * j), C.spacing());

                for (size_t k = 0; k < KK; ++k)
                    ker(A.block(k, KM * i), A.spacing(), true, B.block(k, KN * j), B.spacing(), false);

                ker.store(D.block(i, j), D.spacing());
            }
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_nn(
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
            {
                GemmKernel<T, KM, KN, P> ker;
                ker.load(C.block(KM * i, KN * j), C.spacing());

                for (size_t k = 0; k < KK; ++k)
                    ker(A.block(KM * i, k), A.spacing(), false, B.block(k, KN * j), B.spacing(), false);

                ker.store(D.block(i, j), D.spacing());
            }

        /*
        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                D.store(i, j, C.load(i, j));

        for (size_t i = 0; i < MM; ++i)
            for (size_t k = 0; k < KK; ++k)
            {
                Panel<T, P> A_ik = A.load(i, k);

                for (size_t j = 0; j < NN; ++j)
                {
                    Panel<T, P> p = D.load(i, j);                
                    gemm(A_ik, false, B.load(k, j), false, p);
                    D.store(i, j, p);
                }
            }
        */
    }


    // template <typename T>
    // inline void gemm_tn(
    //     StaticMatrix<T, 8, 8, 4> const& A, StaticMatrix<T, 8, 8, 4> const& B, 
    //     StaticMatrix<T, 8, 8, 4> const& C, StaticMatrix<T, 8, 8, 4>& D)
    // {
    //     size_t const MM = 2;
    //     size_t const NN = 2;
    //     size_t const KK = 2;

    //     Panel<T, 4> p;
        
    //     p = C.load(0, 0);
    //     gemm(A.load(0, 0), true, B.load(0, 0), false, p);
    //     gemm(A.load(1, 0), true, B.load(1, 0), false, p);
    //     D.store(0, 0, p);

    //     p = C.load(0, 1);
    //     gemm(A.load(0, 0), true, B.load(0, 1), false, p);
    //     gemm(A.load(1, 0), true, B.load(1, 1), false, p);
    //     D.store(0, 1, p);

    //     p = C.load(1, 0);
    //     gemm(A.load(0, 1), true, B.load(0, 0), false, p);
    //     gemm(A.load(1, 1), true, B.load(1, 0), false, p);
    //     D.store(1, 0, p);

    //     p = C.load(1, 1);
    //     gemm(A.load(0, 1), true, B.load(0, 1), false, p);
    //     gemm(A.load(1, 1), true, B.load(1, 1), false, p);
    //     D.store(1, 1, p);
    // }
}