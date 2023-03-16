// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/SIMDTrait.h>


namespace blast
{
    using namespace blaze;


    template <typename T>
    struct TileSize;


    template <>
    struct TileSize<double>
    {
        static size_t constexpr value = 4;
    };


    template <>
    struct TileSize<float>
    {
        static size_t constexpr value = 8;
    };


    template <typename T>
    size_t constexpr TileSize_v = TileSize<T>::value;
}
