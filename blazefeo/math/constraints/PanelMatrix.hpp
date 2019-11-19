#pragma once

#include <blazefeo/math/typetraits/IsPanelMatrix.hpp>


namespace blazefeo
{

    //*************************************************************************************************
    /*!\brief Constraint on the data type.
    // \ingroup math_constraints
    //
    // In case the given data type \a T is not a panel matrix type, a compilation
    // error is created.
    */
    #define BLAZEFEO_CONSTRAINT_MUST_BE_PANEL_MATRIX_TYPE(T) \
    static_assert(::blazefeo::IsPanelMatrix_v<T>, "Non-panel matrix type detected")
    //*************************************************************************************************


    //*************************************************************************************************
    /*!\brief Constraint on the data type.
    // \ingroup math_constraints
    //
    // In case the given data type \a T is a panel matrix type, a compilation
    // error is created.
    */
    #define BLAZEFEO_CONSTRAINT_MUST_NOT_BE_PANEL_MATRIX_TYPE(T) \
    static_assert(!::blazefeo::IsPanelMatrix_v<T>, "Panel matrix type detected")
    //*************************************************************************************************
}