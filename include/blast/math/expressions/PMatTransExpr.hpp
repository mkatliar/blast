// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/constraints/PanelMatrix.hpp>
#include <blast/math/expressions/PanelMatrix.hpp>

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/MatTransExpr.h>
#include <blaze/math/expressions/Transformation.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/GetMemberType.h>


namespace blast
{
    //*************************************************************************************************
    /*!\brief Expression object for panel matrix transpositions.
    // \ingroup panel_matrix_expression
    //
    // The PMatTransExpr class represents the compile time expression for transpositions of
    // panel matrices.
    */
    template <typename MT, bool SO>
    class PMatTransExpr
    :   public MatTransExpr< PanelMatrix< PMatTransExpr<MT,SO>, SO > >
    ,   private If_t< IsComputation_v<MT>, Computation, Transformation >
    {
    public:
        //**Type definitions****************************************************************************
        using This          = PMatTransExpr<MT, SO>;        //!< Type of this PMatTransExpr instance.
        using BaseType      = PanelMatrix<This, SO>;        //!< Base type of this PMatTransExpr instance.
        using ResultType    = TransposeType_t<MT>;         //!< Result type for expression template evaluations.
        using OppositeType  = OppositeType_t<ResultType>;  //!< Result type with opposite storage order for expression template evaluations.
        using TransposeType = ResultType_t<MT>;            //!< Transpose type for expression template evaluations.
        using ElementType   = ElementType_t<MT>;           //!< Resulting element type.
        using ReturnType    = ReturnType_t<MT>;            //!< Return type for expression template evaluations.

        //! Data type for composite expression templates.
        using CompositeType = const PMatTransExpr&;

        //! Iterator over the elements of the panel matrix.
        // using ConstIterator = GetConstIterator_t<MT>;

        //! Composite data type of the panel matrix expression.
        using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;
        //**********************************************************************************************

        explicit PMatTransExpr( const MT& pm ) noexcept
            : pm_( pm )
        {}


        //**Access operator*****************************************************************************
        /*!\brief 2D-access to the matrix elements.
        //
        // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
        // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
        // \return The resulting value.
        */
        ReturnType operator()(size_t i, size_t j) const
        {
            BLAZE_INTERNAL_ASSERT( i < pm_.columns(), "Invalid row access index"    );
            BLAZE_INTERNAL_ASSERT( j < pm_.rows()   , "Invalid column access index" );
            return pm_(j, i);
        }


        //**At function*********************************************************************************
        /*!\brief Checked access to the matrix elements.
        //
        // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
        // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
        // \return The resulting value.
        // \exception std::out_of_range Invalid matrix access index.
        */
        ReturnType at(size_t i, size_t j) const
        {
            if (i >= pm_.columns())
                BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );

            if (j >= pm_.rows())
                BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );

            return (*this)(i, j);
        }


        /*!\brief Returns the current number of rows of the matrix.
        //
        // \return The number of rows of the matrix.
        */
        size_t rows() const noexcept
        {
            return pm_.columns();
        }


        /*!\brief Returns the current number of columns of the matrix.
        //
        // \return The number of columns of the matrix.
        */
        size_t columns() const noexcept
        {
            return pm_.rows();
        }


        /*!\brief Returns the spacing between the beginning of two panels.
        //
        // \return The spacing between the beginning of two panels.
        */
        inline size_t spacing() const noexcept
        {
            return pm_.spacing();
        }


        /*!\brief Returns the panel matrix operand.
        //
        // \return The panel matrix operand.
        */
        inline Operand operand() const noexcept
        {
            return pm_;
        }

        /*!\brief Returns whether the expression can alias with the given address \a alias.
        //
        // \param alias The alias to be checked.
        // \return \a true in case the expression can alias, \a false otherwise.
        */
        template< typename T >
        inline bool canAlias( const T* alias ) const noexcept
        {
            return pm_.isAliased( alias );
        }


        /*!\brief Returns whether the expression is aliased with the given address \a alias.
        //
        // \param alias The alias to be checked.
        // \return \a true in case an alias effect is detected, \a false otherwise.
        */
        template< typename T >
        inline bool isAliased( const T* alias ) const noexcept
        {
            return pm_.isAliased( alias );
        }


    private:
        Operand pm_;  //!< Panel matrix of the transposition expression.


        BLAST_CONSTRAINT_MUST_BE_PANEL_MATRIX_TYPE(MT);
        BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER(MT, !SO);
    };


    //=================================================================================================
    //
    //  GLOBAL OPERATORS
    //
    //=================================================================================================

    //*************************************************************************************************
    /*!\brief Calculation of the transpose of the given panel matrix.
    // \ingroup panel_matrix
    //
    // \param pm The panel matrix to be transposed.
    // \return The transpose of the matrix.
    //
    // This function returns an expression representing the transpose of the given panel matrix:

    \code
    using blaze::rowMajor;
    using blaze::columnMajor;

    blaze::DynamicPanelMatrix<double,rowMajor> A;
    blaze::DynamicPanelMatrix<double,columnMajor> B;
    // ... Resizing and initialization
    B = trans( A );
    \endcode
    */
    template <typename MT, bool SO>
    inline decltype(auto) trans(PanelMatrix<MT, SO> const& pm)
    {
        BLAZE_FUNCTION_TRACE;

        using ReturnType = const PMatTransExpr<MT, !SO>;
        return ReturnType(*pm);
    }
}
