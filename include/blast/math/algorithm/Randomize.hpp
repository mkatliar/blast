// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>

#include <array>
#include <vector>
#include <random>


namespace blast
{
    namespace detail
    {
        inline auto& randomEngine()
        {
            static std::mt19937 engine;
            return engine;
        }
    }


    template <typename T>
    requires std::is_floating_point_v<T>
    inline void randomize(T& a)
    {
        std::uniform_real_distribution<T> dist;
        a = dist(detail::randomEngine());
    }


    template <typename T, std::size_t N>
    inline void randomize(std::array<T, N>& a)
    {
        for (T& v : a)
            randomize(v);
    }


    template <typename T>
    inline void randomize(std::vector<T>& a)
    {
        for (T& v : a)
            randomize(v);
    }


    template <Matrix M>
    inline void randomize(M& m) noexcept
    {
        for (size_t i = 0; i < rows(m); ++i)
            for (size_t j = 0; j < columns(m); ++j)
                randomize(m(i, j));
    }


    template <Matrix M>
    requires IsView_v<M>
    inline void randomize(M&& m) noexcept
    {
        randomize(m);
    }
}
