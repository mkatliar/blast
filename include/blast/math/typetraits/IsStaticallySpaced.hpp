// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /**
     * @brief Tests if the given vector data type has spacing between elements
     * which is known at compile-time.
     *
     * @tparam VT vector data type
     */
    template <typename VT>
    struct IsStaticallySpaced;


    /**
     * @brief Shortcut for @a IsStaticallySpaced<VT>::value
     *
     * @tparam VT vector data type
     */
    template <typename VT>
    bool constexpr IsStaticallySpaced_v = IsStaticallySpaced<VT>::value;
}
