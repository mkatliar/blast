// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/util/Types.hpp>


namespace blast
{
    /**
     * @brief Deduces memory spacing of the given data type if it is known at compile-time
     * - for row-major matrices, deduced to the memory distance between rows
     * - for column-major matrices, deduced to the memory distance between columns
     * - for panel-major matrices, deduced to memory distance between panels
     * - for vectors, deduced to memory distance between consecutive elements
     *
     * @tparam T data type
     */
    template <typename T>
    struct Spacing;


    /**
     * @brief Shortcut for @a Spacing<MT>::value
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    size_t constexpr Spacing_v = Spacing<MT>::value;
}
