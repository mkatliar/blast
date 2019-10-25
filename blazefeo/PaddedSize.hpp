#pragma once

#include <blazefeo/SizeT.hpp>


namespace blazefeo
{
    inline size_t constexpr paddedSize(size_t m, size_t block_size)
    {
        return (m / block_size + (m % block_size > 0)) * block_size;
    }
}