#pragma once


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blazefeo/math/panel/Forward.hpp>

#include <blaze/util/IntegralConstant.h>


namespace blazefeo
{
    using namespace blaze;

    //=================================================================================================
    //
    //  CLASS DEFINITION
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Auxiliary helper struct for the IsPanelMatrix type trait.
    // \ingroup math_type_traits
    */
    template< typename T >
    struct IsPanelMatrixHelper
    {
    private:
        //**********************************************************************************************
        static T* create();

        template <typename MT, bool SO>
        static TrueType test( const PanelMatrix<MT, SO>* );

        template <typename MT, bool SO>
        static TrueType test( const volatile PanelMatrix<MT, SO>* );

        static FalseType test( ... );
        //**********************************************************************************************

    public:
        //**********************************************************************************************
        using Type = decltype( test( create() ) );
        //**********************************************************************************************
    };
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*!\brief Compile time check for panel matrix types.
    // \ingroup math_type_traits
    //
    // This type trait tests whether or not the given template parameter is a panel, N-dimensional
    // matrix type. In case the type is a panel matrix type, the \a value member constant is set
    // to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
    // \a TrueType. Otherwise \a yes is set to \a false, \a Type is \a FalseType, and the class
    // derives from \a FalseType.

    \code
    blaze::IsPanelMatrix< DynamicPanelMatrix<double,false> >::value     // Evaluates to 1
    blaze::IsPanelMatrix< const DynamicPanelMatrix<float,true> >::Type  // Results in TrueType
    blaze::IsPanelMatrix< volatile DynamicPanelMatrix<int,true> >       // Is derived from TrueType
    blaze::IsPanelMatrix< CompressedMatrix<double,false>::value    // Evaluates to 0
    blaze::IsPanelMatrix< CompressedVector<double,true> >::Type    // Results in FalseType
    blaze::IsPanelMatrix< DynamicVector<double,true> >             // Is derived from FalseType
    \endcode
    */
    template <typename T>
    struct IsPanelMatrix
    :   public IsPanelMatrixHelper<T>::Type
    {};
    //*************************************************************************************************


    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    /*!\brief Specialization of the IsPanelMatrix type trait for references.
    // \ingroup math_type_traits
    */
    template <typename T>
    struct IsPanelMatrix<T&>
    :   public FalseType
    {};
    /*! \endcond */
    //*************************************************************************************************


    //*************************************************************************************************
    /*!\brief Auxiliary variable template for the IsPanelMatrix type trait.
    // \ingroup math_type_traits
    //
    // The IsPanelMatrix_v variable template provides a convenient shortcut to access the nested
    // \a value of the IsPanelMatrix class template. For instance, given the type \a T the
    // following two statements are identical:

    \code
    constexpr bool value1 = blaze::IsPanelMatrix<T>::value;
    constexpr bool value2 = blaze::IsPanelMatrix_v<T>;
    \endcode
    */
    template< typename T >
    constexpr bool IsPanelMatrix_v = IsPanelMatrix<T>::value;
    //*************************************************************************************************
}
