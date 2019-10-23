#pragma once

#include <smoke/SizeT.hpp>


namespace smoke
{
    template <typename T, size_t M, size_t N, size_t BS, bool TA, bool TB>
    class GemmKernel;


    template <typename Ker>
    struct GemmKernelTraits;


    template <typename T, size_t M, size_t N, size_t BS, bool TA, bool TB>
    struct GemmKernelTraits<GemmKernel<T, M, N, BS, TA, TB>>
    {
        static size_t constexpr alignment = GemmKernel<T, M, N, BS, TA, TB>::alignment;
        static size_t constexpr blockSize = BS;
        static size_t constexpr blockRows = M;
        static size_t constexpr blockColumns = N;
        static size_t constexpr rows = M * BS;
        static size_t constexpr columns = N * BS;
        static size_t constexpr elementCount = rows * columns;
        static size_t constexpr blockElementCount = BS * BS;
        static bool constexpr tA = TA;
        static bool constexpr tB = TB;
    };


    template <typename T, size_t M, size_t N, size_t BS, bool TA, bool TB>
    inline void gemm(GemmKernel<T, M, N, BS, TA, TB> ker, size_t K,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd)
    {
        ker.load(c, sc);

        for (size_t k = 0; k < K; ++k)
        {
            ker(a, sa, b, sb);

            a += TA ? M * sa : BS;
            b += TB ? BS : N * sb;
        }

        ker.store(d, sd);
    }


    template <typename T, size_t M, size_t N, size_t BS, bool TA, bool TB>
    inline void gemm(GemmKernel<T, M, N, BS, TA, TB> ker, size_t K,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd,
        size_t md, size_t nd)
    {
        ker.load(c, sc);
        
        for (size_t k = 0; k < K; ++k)
        {
            ker(a, sa, b, sb);

            a += TA ? M * sa : BS;
            b += TB ? BS : N * sb;
        }

        ker.store(d, sd, md, nd);
    }
}