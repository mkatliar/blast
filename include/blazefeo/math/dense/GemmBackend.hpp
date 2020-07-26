#pragma once

#include <blazefeo/math/simd/RegisterMatrix.hpp>


namespace blazefeo
{
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
}