#pragma once

#include <smoke/StaticPanelMatrix.hpp>
#include <blaze/Math.h>

#include <array>


namespace smoke
{
    template <typename T, std::size_t N>
    inline void randomize(std::array<T, N>& a)
    {
        for (T& v : a)
            blaze::randomize(v);
    }
}