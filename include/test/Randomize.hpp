// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/util/Random.h>

#include <array>
#include <vector>


namespace blast
{
    using blaze::randomize;


    template <typename T, std::size_t N>
    inline void randomize(std::array<T, N>& a)
    {
        for (T& v : a)
            blaze::randomize(v);
    }


    template <typename T>
    inline void randomize(std::vector<T>& a)
    {
        for (T& v : a)
            blaze::randomize(v);
    }
}
