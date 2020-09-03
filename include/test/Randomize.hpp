// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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