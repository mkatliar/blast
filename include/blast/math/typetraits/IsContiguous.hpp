// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /**
     * @brief Tests whether the elements of a given data type lie contiguous in memory
     *
     * @tparam T data type
     */
    template <typename T>
    struct IsContiguous;


    /**
     * @brief Shortcut for @a IsContiguous<T>::value
     *
     * @tparam T data type
     */
    template <typename T>
    bool constexpr IsContiguous_v = IsContiguous<T>::value;
}
