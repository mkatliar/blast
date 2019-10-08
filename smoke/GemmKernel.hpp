#pragma once

#include <smoke/SizeT.hpp>


namespace smoke
{
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
}