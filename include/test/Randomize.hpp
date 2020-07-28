#pragma once

#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/Blaze.hpp>

#include <array>


namespace blazefeo
{
    template <typename T, std::size_t N>
    inline void randomize(std::array<T, N>& a)
    {
        for (T& v : a)
            blaze::randomize(v);
    }
}