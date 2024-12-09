// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <cstdlib>
#include <utility>


namespace blast
{
    /**
     * @brief Wraps a compile-time constant of type @a std::size_t in a type.
     *
     * @tparam N the value of the constant.
     */
    template <std::size_t N>
    using SizeConstant = std::integral_constant<std::size_t, N>;
}
