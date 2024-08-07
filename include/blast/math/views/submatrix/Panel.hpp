// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/PanelMatrix.hpp>
#include <blast/math/views/submatrix/BaseTemplate.hpp>
#include <blast/math/constraints/Submatrix.hpp>
#include <blast/math/panel/PanelSize.hpp>
#include <blast/math/constraints/PanelMatrix.hpp>
#include <blast/util/Assert.hpp>

#include <blaze/math/Constraints.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/views/Check.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/Aliases.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Exception.h>
#include <blaze/util/Assert.h>


namespace blast
{
    //=================================================================================================
    //
    //  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED ROW-MAJOR PANEL SUBMATRICES
    //
    //=================================================================================================

    //*************************************************************************************************
    /*!\brief Specialization of PanelSubmatrix for unaligned row-major panel submatrices.
    // \ingroup submatrix
    //
    // This Specialization of PanelSubmatrix adapts the class template to the requirements of unaligned
    // row-major panel submatrices.
    */
    template <typename MT, bool SO>
    class PanelSubmatrix<MT, SO>
    : public View<PanelMatrix<PanelSubmatrix<MT, SO>, SO>>
    {
    private:
        //**Type definitions****************************************************************************
        using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the matrix expression.
        //**********************************************************************************************


    public:
        //**Type definitions****************************************************************************
        //! Type of this PanelSubmatrix instance.
        using This = PanelSubmatrix<MT, SO>;

        using BaseType      = PanelMatrix<This, SO>;       //!< Base type of this PanelSubmatrix instance.
        using ViewedType    = MT;                            //!< The type viewed by this PanelSubmatrix instance.
        // using ResultType    = SubmatrixTrait_t<MT>;  //!< Result type for expression template evaluations.
        // using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
        // using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
        using ElementType   = ElementType_t<MT>;             //!< Type of the submatrix elements.
        // using SIMDType      = SIMDTrait_t<ElementType>;      //!< SIMD type of the submatrix elements.
        using ReturnType    = ReturnType_t<MT>;              //!< Return type for expression template evaluations
        using CompositeType = const PanelSubmatrix&;              //!< Data type for composite expression templates.

        //! Reference to a constant submatrix value.
        using ConstReference = ConstReference_t<MT>;

        //! Reference to a non-constant submatrix value.
        using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;

        //! Pointer to a constant submatrix value.
        using ConstPointer = ConstPointer_t<MT>;

        //! Pointer to a non-constant submatrix value.
        using Pointer = If_t< IsConst_v<MT> || !HasMutableDataAccess_v<MT>, ConstPointer, Pointer_t<MT> >;
        //**********************************************************************************************


        //**Constructors********************************************************************************
        template <typename... RSAs>
        explicit inline constexpr PanelSubmatrix(MT& matrix, size_t i, size_t j, size_t m, size_t n, RSAs... args)
        :   matrix_   ( matrix  )  // The matrix containing the submatrix
        ,   i_(i)
        ,   j_(j)
        ,   m_(m)
        ,   n_(n)
        ,   data_(&matrix_(row(), column()))
        {
            if( !Contains_v< TypeList<RSAs...>, Unchecked > )
            {
                if( ( row() + rows() > matrix_.rows() ) || ( column() + columns() > matrix_.columns() ) ) {
                    BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
                }

                if (IsRowMajorMatrix_v<MT> && column() % panelSize_ > 0)
                    BLAZE_THROW_LOGIC_ERROR("Submatrices of a row-major panel matrix which are not horizontally aligned on panel boundary "
                        "are currently not supported");

                if (IsColumnMajorMatrix_v<MT> && row() % panelSize_ > 0)
                    BLAZE_THROW_LOGIC_ERROR("Submatrices of a column-major panel matrix which are not vertically aligned on panel boundary "
                        "are currently not supported");
            }
            else
            {
                BLAST_USER_ASSERT( row()    + rows()    <= matrix_.rows()   , "Invalid submatrix specification" );
                BLAST_USER_ASSERT( column() + columns() <= matrix_.columns(), "Invalid submatrix specification" );
            }
        }


        PanelSubmatrix( const PanelSubmatrix& ) = default;


        //=================================================================================================
        //
        //  UTILITY FUNCTIONS
        //
        //=================================================================================================
        size_t constexpr row() const noexcept
        {
            return i_;
        };


        size_t constexpr column() const noexcept
        {
            return j_;
        };


        size_t constexpr rows() const noexcept
        {
            return m_;
        };


        size_t constexpr columns() const noexcept
        {
            return n_;
        };


        /// @brief Offset of the first matrix element from the start of the panel.
        ///
        /// In rows for column-major matrices, in columns for row-major matrices.
        size_t constexpr offset() const
        {
            return SO == (columnMajor ? i_ : j_) % panelSize_;
        }


        MT& operand() noexcept
        {
            return matrix_;
        }


        const MT& operand() const noexcept
        {
            return matrix_;
        }


        size_t spacing() const noexcept
        {
            return matrix_.spacing();
        }


        size_t capacity() const noexcept
        {
            return rows() * columns();
        }


        //=================================================================================================
        //
        //  DATA ACCESS FUNCTIONS
        //
        //=================================================================================================

        Reference operator()( size_t i, size_t j )
        {
            BLAST_USER_ASSERT( i < rows()   , "Invalid row access index"    );
            BLAST_USER_ASSERT( j < columns(), "Invalid column access index" );

            return matrix_(row()+i,column()+j);
        }


        ConstReference operator()( size_t i, size_t j ) const
        {
            BLAST_USER_ASSERT( i < rows()   , "Invalid row access index"    );
            BLAST_USER_ASSERT( j < columns(), "Invalid column access index" );

            return const_cast<const MT&>( matrix_ )(row()+i, column()+j);
        }


        Reference at( size_t i, size_t j )
        {
            if( i >= rows() ) {
                BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
            }
            if( j >= columns() ) {
                BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
            }
            return (*this)(i,j);
        }


        ConstReference at( size_t i, size_t j ) const
        {
            if( i >= rows() ) {
                BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
            }
            if( j >= columns() ) {
                BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
            }
            return (*this)(i,j);
        }


        Pointer data() noexcept
        {
            return data_;
        }


        ConstPointer data() const noexcept
        {
            return data_;
        }


    private:
        static size_t constexpr panelSize_ = PanelSize_v<ElementType>;

        Operand matrix_;        //!< The matrix containing the submatrix.

        size_t const i_;
        size_t const j_;
        size_t const m_;
        size_t const n_;

        // Pointer to the first element of the submatrix
        Pointer const data_;


        //**Friend declarations*************************************************************************
        template< typename MT2, bool SO2, size_t... CSAs2 > friend class PanelSubmatrix;
        //**********************************************************************************************

        //**Compile time checks*************************************************************************
        BLAST_CONSTRAINT_MUST_BE_PANEL_MATRIX_TYPE    ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE   ( MT );
        // BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( MT );
        BLAST_CONSTRAINT_MUST_NOT_BE_PANEL_SUBMATRIX_TYPE(MT);
        //**********************************************************************************************
    };


    template <typename MT, bool SO>
    inline decltype(auto) submatrix(PanelSubmatrix<MT, SO> const& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT const, SO>(matrix.operand(), matrix.row() + row, matrix.column() + column, m, n);
    }


    template <typename MT, bool SO>
    inline decltype(auto) submatrix(PanelSubmatrix<MT, SO>& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT, SO>(matrix.operand(), matrix.row() + row, matrix.column() + column, m, n);
    }
}
