// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blast/math/typetraits/IsPanelSubmatrix.hpp>


namespace blast
{
    //=================================================================================================
    //
    //  MUST_BE_SUBMATRIX_TYPE CONSTRAINT
    //
    //=================================================================================================

    //*************************************************************************************************
    /*!\brief Constraint on the data type.
    // \ingroup math_constraints
    //
    // In case the given data type \a T is not a submatrix type (i.e. a dense or sparse submatrix),
    // a compilation error is created.
    */
    #define BLAST_CONSTRAINT_MUST_BE_PANEL_SUBMATRIX_TYPE(T) \
    static_assert( ::blast::IsPanelSubmatrix_v<T>, "Non-PanelSubmatrix type detected" )
    //*************************************************************************************************




    //=================================================================================================
    //
    //  MUST_NOT_BE_SUBMATRIX_TYPE CONSTRAINT
    //
    //=================================================================================================

    //*************************************************************************************************
    /*!\brief Constraint on the data type.
    // \ingroup math_constraints
    //
    // In case the given data type \a T is a submatrix type (i.e. a dense or sparse submatrix), a
    // compilation error is created.
    */
    #define BLAST_CONSTRAINT_MUST_NOT_BE_PANEL_SUBMATRIX_TYPE(T) \
    static_assert( !::blast::IsPanelSubmatrix_v<T>, "PanelSubmatrix type detected" )
    //*************************************************************************************************

}
