// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/IsPanelMatrix.hpp>


namespace blast
{

    //*************************************************************************************************
    /*!\brief Constraint on the data type.
    // \ingroup math_constraints
    //
    // In case the given data type \a T is not a panel matrix type, a compilation
    // error is created.
    */
    #define BLAST_CONSTRAINT_MUST_BE_PANEL_MATRIX_TYPE(T) \
    static_assert(::blast::IsPanelMatrix_v<T>, "Non-panel matrix type detected")
    //*************************************************************************************************


    //*************************************************************************************************
    /*!\brief Constraint on the data type.
    // \ingroup math_constraints
    //
    // In case the given data type \a T is a panel matrix type, a compilation
    // error is created.
    */
    #define BLAST_CONSTRAINT_MUST_NOT_BE_PANEL_MATRIX_TYPE(T) \
    static_assert(!::blast::IsPanelMatrix_v<T>, "Panel matrix type detected")
    //*************************************************************************************************
}