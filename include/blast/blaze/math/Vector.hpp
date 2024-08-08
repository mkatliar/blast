// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/blaze/math/typetraits/IsContiguous.hpp>

#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsView.h>


namespace blaze
{
    /**
     * @brief Memory distance between consecutive elements of a dense Blaze vector
     *
     * NOTE: The function is declared in blaze namespace s.t. it can be found by ADL.
     *
     * @tparam VT vector type
     *
     * @param v vector
     *
     * @return memory distance between consecutive elements of @a v
     */
    template <typename VT>
    requires blaze::IsDenseVector_v<VT>
    inline size_t spacing(VT const& v) noexcept
    {
        if constexpr (IsContiguous_v<VT>)
            return 1;
        else
            if constexpr (blaze::IsView_v<VT>)
                return spacing(v.operand());
            else
                static_assert(false, "Spacing is not defined for a type which is not a view and is not contiguous");
    }
}
