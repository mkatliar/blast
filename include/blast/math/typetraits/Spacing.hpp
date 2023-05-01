// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/math/Submatrix.h>

#include <blaze/math/typetraits/IsSubmatrix.h>
#include <blaze/math/typetraits/IsView.h>
#include <cstddef>


namespace blast
{
    template <typename MT>
    struct Spacing
    {
        static std::size_t constexpr value = MT::spacing();
    };


    template <typename MT, blaze::AlignmentFlag AF, bool SO, bool DF, std::size_t... CSAs>
    struct Spacing<blaze::Submatrix<MT, AF, SO, DF, CSAs...>>
    :   Spacing<MT>
    {
    };


    template <typename MT, bool SO, bool DF>
    struct Spacing<blaze::LowerMatrix<MT, SO, DF>>
    :   Spacing<MT>
    {
    };


    template <typename MT>
    std::size_t constexpr Spacing_v = Spacing<MT>::value;
}