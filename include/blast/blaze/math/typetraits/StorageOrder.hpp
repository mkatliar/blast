// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/StorageOrder.hpp>

#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/StorageOrder.h>

#include <type_traits>


namespace blast
{
    /**
     * @brief Specialization for Blaze matrices
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    requires blaze::IsDenseMatrix_v<MT>
    struct StorageOrderHelper<MT>
    :   std::integral_constant<StorageOrder,
            blaze::StorageOrder_v<MT> == blaze::columnMajor ? columnMajor : rowMajor
        >
    {};
}
