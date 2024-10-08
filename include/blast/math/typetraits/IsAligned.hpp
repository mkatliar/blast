// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /**
     * @brief Tests if the given matrix or vector type storage is aligned.
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsAligned;


    /**
     * @brief Specialization for const types
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsAligned<T const> : IsAligned<T> {};


    /**
     * @brief Shortcut for @a IsAligned<T>::value
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    bool constexpr IsAligned_v = IsAligned<T>::value;
}
