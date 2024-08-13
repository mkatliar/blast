// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/StorageOrder.hpp>


namespace blast
{
    /**
     * @brief Deduces storage order of a matrix or a matrix pointer type
     *
     * NODE: the naming is not consistent here, because we already have @a StorageOrder enum.
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    struct StorageOrderHelper;


    /**
     * @brief Specialization for const types
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    struct StorageOrderHelper<MT const> : StorageOrderHelper<MT> {};


    /**
     * @brief Shortcut for @a StorageOrderHelper<MT>::value
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    StorageOrder constexpr StorageOrder_v = StorageOrderHelper<MT>::value;
}
