#pragma once

#include <blazefeo/math/views/Forward.hpp>

#include <blaze/util/IntegralConstant.h>


namespace blazefeo
{
    template< typename T >
    struct IsPanelSubmatrix
    : public FalseType
    {};
    

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the IsPanelSubmatrix type trait for 'Submatrix'.
    // \ingroup math_type_traits
    */
    template< typename MT, bool SO, size_t... CSAs >
    struct IsPanelSubmatrix< PanelSubmatrix<MT,SO,CSAs...> >
    : public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the IsPanelSubmatrix type trait for 'const Submatrix'.
    // \ingroup math_type_traits
    */
    template< typename MT, bool SO, size_t... CSAs >
    struct IsPanelSubmatrix< const PanelSubmatrix<MT,SO,CSAs...> >
    : public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the IsPanelSubmatrix type trait for 'volatile Submatrix'.
    // \ingroup math_type_traits
    */
    template< typename MT, bool SO, size_t... CSAs >
    struct IsPanelSubmatrix< volatile PanelSubmatrix<MT,SO,CSAs...> >
    : public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the IsPanelSubmatrix type trait for 'const volatile Submatrix'.
    // \ingroup math_type_traits
    */
    template< typename MT, bool SO, size_t... CSAs >
    struct IsPanelSubmatrix< const volatile PanelSubmatrix<MT,SO,CSAs...> >
    : public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*!\brief Auxiliary variable template for the IsPanelSubmatrix type trait.
    // \ingroup math_type_traits
    //
    // The IsPanelSubmatrix_v variable template provides a convenient shortcut to access the nested
    // \a value of the IsPanelSubmatrix class template. For instance, given the type \a T the following
    // two statements are identical:

    \code
    constexpr bool value1 = blaze::IsPanelSubmatrix<T>::value;
    constexpr bool value2 = blaze::IsPanelSubmatrix_v<T>;
    \endcode
    */
    template< typename T >
    constexpr bool IsPanelSubmatrix_v = IsPanelSubmatrix<T>::value;
    //*************************************************************************************************

} // namespace blaze
