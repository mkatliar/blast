// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /**
     * @brief Tests if dimensions of the given matrix or vector type are known at compile-time.
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsStatic;


    /**
     * @brief Specialization for const types
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsStatic<T const> : IsStatic<T> {};


    /**
     * @brief Shortcut for @a IsStatic<T>::value
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    bool constexpr IsStatic_v = IsStatic<T>::value;
}
