#pragma once


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blazefeo/math/typetraits/IsPanelSubmatrix.hpp>


namespace blazefeo
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
    #define BLAZEFEO_CONSTRAINT_MUST_BE_PANEL_SUBMATRIX_TYPE(T) \
    static_assert( ::blazefeo::IsPanelSubmatrix_v<T>, "Non-PanelSubmatrix type detected" )
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
    #define BLAZEFEO_CONSTRAINT_MUST_NOT_BE_PANEL_SUBMATRIX_TYPE(T) \
    static_assert( !::blazefeo::IsPanelSubmatrix_v<T>, "PanelSubmatrix type detected" )
    //*************************************************************************************************

}
