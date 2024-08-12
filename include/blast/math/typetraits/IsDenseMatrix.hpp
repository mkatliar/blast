// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <type_traits>

namespace blast
{
    /**
     * @brief Tests if the given data type is a dense matrix with column- or row-major storage
     *
     * @tparam T data type
     */
    template <typename T>
    struct IsDenseMatrix : std::false_type {};


    /**
     * @brief Shortcut for @a IsDenseMatrix<T>::value
     *
     * @tparam T data type
     */
    template <typename T>
    bool constexpr IsDenseMatrix_v = IsDenseMatrix<T>::value;
}
