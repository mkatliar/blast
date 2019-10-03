#pragma once

#include <smoke/SizeT.hpp>


namespace smoke
{
    inline size_t constexpr paddedSize(size_t m, size_t n)
    {
        return (m / n + (m % n > 0)) * n;
    }
}