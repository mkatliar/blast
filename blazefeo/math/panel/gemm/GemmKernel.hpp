#pragma once

#include <blaze/util/Types.h>
#include <blaze/system/Inline.h>


namespace blazefeo
{
    using namespace blaze;

    
    template <typename T, size_t M, size_t N, size_t BS>
    class GemmKernel;


    template <typename Ker>
    struct GemmKernelTraits;


    template <typename T, size_t M, size_t N, size_t BS>
    struct GemmKernelTraits<GemmKernel<T, M, N, BS>>
    {
        static size_t constexpr alignment = GemmKernel<T, M, N, BS>::alignment;
        static size_t constexpr blockSize = BS;
        static size_t constexpr blockRows = M;
        static size_t constexpr blockColumns = N;
        static size_t constexpr rows = M * BS;
        static size_t constexpr columns = N * BS;
        static size_t constexpr elementCount = rows * columns;
        static size_t constexpr blockElementCount = BS * BS;
    };


    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void gemm(GemmKernel<T, M, N, BS>& ker, T const * a, size_t sa, T const * b, size_t sb)
    {
        ker.template gemm<TA, TB>(a, sa, b, sb);
    }


    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void gemm(GemmKernel<T, M, N, BS>& ker, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        ker.template gemm<TA, TB>(a, sa, b, sb, m, n);
    }


    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void gemm(GemmKernel<T, M, N, BS> ker, size_t K,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd)
    {
        ker.load(c, sc);

        for (size_t k = 0; k < K; ++k)
        {
            gemm<TA, TB>(ker, a, sa, b, sb);

            a += TA ? M * sa : BS;
            b += TB ? BS : N * sb;
        }

        ker.store(d, sd);
    }


    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t BS>
    BLAZE_ALWAYS_INLINE void gemm(GemmKernel<T, M, N, BS> ker, size_t K,
        T const * a, size_t sa, T const * b, size_t sb, T const * c, size_t sc, T * d, size_t sd,
        size_t md, size_t nd)
    {
        ker.load(c, sc, md, nd);
        
        for (size_t k = 0; k < K; ++k)
        {
            gemm<TA, TB>(ker, a, sa, b, sb, md, nd);

            a += TA ? M * sa : BS;
            b += TB ? BS : N * sb;
        }

        ker.store(d, sd, md, nd);
    }
}