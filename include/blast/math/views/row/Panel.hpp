// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/PanelMatrix.hpp>
#include <blast/math/views/submatrix/BaseTemplate.hpp>
#include <blast/math/constraints/Submatrix.hpp>
#include <blast/math/panel/PanelSize.hpp>
#include <blast/math/constraints/PanelMatrix.hpp>
#include <blast/math/TypeTraits.hpp>


namespace blast
{
    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    class PanelMatrixRow
    {
    private:
        //**Type definitions****************************************************************************
        using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the matrix expression.
        //**********************************************************************************************


    public:
        using ViewedType    = MT;                            //!< The type viewed by this PanelMatrixRow instance.
        using ElementType   = ElementType_t<MT>;             //!< Type of the submatrix elements.
        // using SIMDType      = SIMDTrait_t<ElementType>;      //!< SIMD type of the submatrix elements.
        using ReturnType    = ReturnType_t<MT>;              //!< Return type for expression template evaluations
        using CompositeType = const PanelMatrixRow&;              //!< Data type for composite expression templates.

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
        template <typename... RRAs>
        explicit inline constexpr PanelMatrixRow(MT& matrix, size_t i, RRAs... args)
        :   matrix_(matrix)
        ,   i_(i)
        {
            if (!Contains_v<TypeList<RRAs...>, Unchecked>)
            {
                if (matrix_.rows() <= row())
                    BLAZE_THROW_INVALID_ARGUMENT("Invalid row access index");
            }
            else
            {
                BLAST_USER_ASSERT(row() < matrix_.rows(), "Invalid row access index");
            }
        }


        PanelMatrixRow( const PanelMatrixRow& ) = default;


        //=================================================================================================
        //
        //  UTILITY FUNCTIONS
        //
        //=================================================================================================
        size_t constexpr row() const noexcept
        {
            return i_;
        };


        size_t constexpr size() const noexcept
        {
            return matrix_.columns();
        };


        /// @brief Offset of the first row element from the start of the panel.
        size_t constexpr offset() const
        {
            return SO == rowMajor ? matrix_.offset() : 0;
        }


        MT& operand() noexcept
        {
            return matrix_;
        }


        const MT& operand() const noexcept
        {
            return matrix_;
        }


        // size_t spacing() const noexcept
        // {
        //     return matrix_.spacing();
        // }


        //=================================================================================================
        //
        //  DATA ACCESS FUNCTIONS
        //
        //=================================================================================================

        Reference operator[](size_t j)
        {
            BLAST_USER_ASSERT(j < size(), "Invalid column access index");
            return matrix_(row(), j);
        }


        ConstReference operator[](size_t j) const
        {
            BLAST_USER_ASSERT(j < size(), "Invalid column access index");
            return const_cast<const MT&>(matrix_)(row(), j);
        }


        template <size_t SS>
        auto load(size_t j) const
        {
            static_assert(SO == rowMajor, "load() is not implemented on rows of column-major panel matrices");

            BLAST_USER_ASSERT(j < size(), "Invalid column access index" );
            return matrix_.template load<SS>(row(), j);
        }


        Reference at(size_t j)
        {
            if (j >= size())
                BLAZE_THROW_OUT_OF_RANGE("Invalid column access index");

            return (*this)(row(), j);
        }


        ConstReference at(size_t j) const
        {
            if (j >= size())
                BLAZE_THROW_OUT_OF_RANGE("Invalid column access index");

            return (*this)(row(), j);
        }


    private:
        static size_t constexpr panelSize_ = PanelSize_v<ElementType>;

        Operand matrix_;
        size_t const i_;


        //**Friend declarations*************************************************************************
        template< typename MT2, bool SO2, size_t... CSAs2 > friend class PanelMatrixRow;
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
}
