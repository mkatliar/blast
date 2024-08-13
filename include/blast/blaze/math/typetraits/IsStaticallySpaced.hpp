// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/IsStaticallySpaced.hpp>
#include <blast/math/typetraits/IsContiguous.hpp>

#include <blaze/math/typetraits/IsVector.h>
#include <blaze/math/typetraits/IsView.h>
#include <blaze/math/typetraits/IsStatic.h>
#include <blaze/math/Aliases.h>


namespace blast
{
    /**
     * @brief Specialization for Blaze vectors which are not views
     *
     * Blaze vectors which are not views are statically spaced iff they are contiguous
     *
     * @tparam VT vector type
     */
    template <typename VT>
    requires blaze::IsVector_v<VT> && (!blaze::IsView_v<VT>)
    struct IsStaticallySpaced<VT> : IsContiguous<VT> {};


    /**
     * @brief Specialization for Blaze vectors which are views
     *
     * Blaze vectors which are views are statically spaced iff they are contiguous or their viewed type is static
     *
     * @tparam VT vector type
     */
    template <typename VT>
    requires blaze::IsVector_v<VT> && blaze::IsView_v<VT>
    struct IsStaticallySpaced<VT>
    :   std::integral_constant<bool,
            IsContiguous_v<VT> || blaze::IsStatic_v<blaze::ViewedType_t<VT>>
        >
    {};
}
