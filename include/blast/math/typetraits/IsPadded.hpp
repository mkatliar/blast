// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /**
     * @brief Tests if the given matrix or vector type storage is padded.
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsPadded;


    /**
     * @brief Specialization for const types
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    struct IsPadded<T const> : IsPadded<T> {};


    /**
     * @brief Shortcut for @a IsPadded<T>::value
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    bool constexpr IsPadded_v = IsPadded<T>::value;
}
