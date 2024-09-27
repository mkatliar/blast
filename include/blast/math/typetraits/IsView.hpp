// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <type_traits>


namespace blast
{
    /**
     * @brief Tests if the given matrix or vector type is a view.
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsView : std::integral_constant<bool, false> {};


    /**
     * @brief Shortcut for @a IsView<T>::value
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    bool constexpr IsView_v = IsView<T>::value;
}
