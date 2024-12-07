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
    {
    private:
        using Operand  = MT&;

    public:
        using This = PanelSubmatrix<MT, SO>;
        using ViewedType    = MT;                            //!< The type viewed by this PanelSubmatrix instance.
        using ElementType   = ElementType_t<MT>;             //!< Type of the submatrix elements.

        //! Reference to a constant submatrix value.
        using ConstReference = ElementType const&;

        //! Reference to a non-constant submatrix value.
        using Reference = std::conditional<std::is_const_v<MT>, ElementType const&, ElementType&>;

        //! Pointer to a constant submatrix value.
        using ConstPointer = ElementType const *;

        //! Pointer to a non-constant submatrix value.
        using Pointer = std::conditional<std::is_const_v<MT>, ElementType const *, ElementType *>;
        //**********************************************************************************************


        //**Constructors********************************************************************************
        explicit inline constexpr PanelSubmatrix(MT& matrix, size_t i, size_t j, size_t m, size_t n)
        :   matrix_   ( matrix  )  // The matrix containing the submatrix
        ,   i_(i)
        ,   j_(j)
        ,   m_(m)
        ,   n_(n)
        ,   data_(&matrix_(row(), column()))
        {
            BLAST_USER_ASSERT( row()    + rows()    <= matrix_.rows()   , "Invalid submatrix specification" );
            BLAST_USER_ASSERT( column() + columns() <= matrix_.columns(), "Invalid submatrix specification" );
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

        Reference operator()(size_t i, size_t j)
        {
            BLAST_USER_ASSERT( i < rows()   , "Invalid row access index"    );
            BLAST_USER_ASSERT( j < columns(), "Invalid column access index" );

            return matrix_(row()+i, column()+j);
        }


        ConstReference operator()(size_t i, size_t j) const
        {
            BLAST_USER_ASSERT( i < rows()   , "Invalid row access index"    );
            BLAST_USER_ASSERT( j < columns(), "Invalid column access index" );

            return const_cast<const MT&>( matrix_ )(row()+i, column()+j);
        }


        Reference at(size_t i, size_t j)
        {
            if (i >= rows())
                throw std::out_of_range {"Invalid row access index"};

            if (j >= columns())
                throw std::out_of_range {"Invalid column access index"};

            return (*this)(i, j);
        }


        ConstReference at(size_t i, size_t j) const
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
