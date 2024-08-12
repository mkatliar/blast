// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/IsContiguous.hpp>

#include <blaze/math/Aliases.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/math/typetraits/IsTransExpr.h>

#include <type_traits>


namespace blast
{
    /**
     * @brief Specialization for Blaze vectors which are not transpose expressions
     *
     * @tparam T type
     */
    template <typename T>
    requires blaze::IsVector_v<T> && (!blaze::IsTransExpr_v<T>)
    struct IsContiguous<T> : blaze::IsContiguous<T> {};


    /**
     * @brief Specialization for Blaze vector transpose expressions
     *
     * The trasnposed vector expression is contiguous iff its operand is contiguous.
     *
     * This specialization is required to fix this Blaze bug: https://bitbucket.org/blaze-lib/blaze/issues/474
     *
     * @tparam T type
     */
    template <typename T>
    requires blaze::IsVector_v<T> && blaze::IsTransExpr_v<T>
    struct IsContiguous<T> : IsContiguous<std::remove_reference_t<blaze::Operand_t<T>>> {};
}
