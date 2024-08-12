// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/Spacing.hpp>
#include <blast/blaze/math/typetraits/IsContiguous.hpp>

#include <blaze/math/Aliases.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsTransExpr.h>
#include <blaze/math/typetraits/IsView.h>

#include <type_traits>


namespace blast
{
    /**
     * @brief Specialization for Blaze dense matrices which are not transpose expressions
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    requires blaze::IsDenseMatrix_v<MT> && (!blaze::IsTransExpr_v<MT>)
    struct Spacing<MT> : std::integral_constant<size_t, MT::spacing()> {};


    /**
     * @brief Specialization for Blaze dense matrix transpose expressions
     *
     * @tparam MT matrix type
     */
    template <typename MT>
    requires blaze::IsDenseMatrix_v<MT> && blaze::IsTransExpr_v<MT>
    struct Spacing<MT> : Spacing<std::remove_reference_t<blaze::Operand_t<MT>>> {};


    /**
     * @brief Specialization for contiguous dense Blaze vectors
     *
     * @tparam VT vector type
     */
    template <typename VT>
    requires blaze::IsDenseVector_v<VT> && IsContiguous_v<VT>
    struct Spacing<VT> : std::integral_constant<size_t, 1> {};


    /**
     * @brief Specialization for non-contiguous dense Blaze vector views
     *
     * @tparam VT vector type
     */
    template <typename VT>
    requires blaze::IsDenseVector_v<VT> && (!IsContiguous_v<VT>) && blaze::IsView_v<VT>
    struct Spacing<VT> : Spacing<blaze::ViewedType_t<VT>> {};
}
