#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/BaseTemplate.hpp>
#include <blazefeo/math/constraints/Submatrix.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/math/constraints/Submatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>

#include <algorithm>
#include <iterator>


namespace blazefeo
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
    template< typename MT       // Type of the panel matrix
            , bool SO >  // Compile time submatrix arguments
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
        using ResultType    = SubmatrixTrait_t<MT>;  //!< Result type for expression template evaluations.
        using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
        using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
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
        ,   data_(matrix_.tile(row() / tileSize_, 0) + column() * tileSize_)
        {
            if( !Contains_v< TypeList<RSAs...>, Unchecked > )
            {
                if( ( row() + rows() > matrix_.rows() ) || ( column() + columns() > matrix_.columns() ) ) {
                    BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
                }

                if (IsRowMajorMatrix_v<MT> && row() % tileSize_ > 0)
                    BLAZE_THROW_LOGIC_ERROR("Submatrices of a row-major panel matrix which are not vertically aligned on a tile boundary "
                        "are currently not supported");

                if (IsColumnMajorMatrix_v<MT> && column() % tileSize_ > 0)
                    BLAZE_THROW_LOGIC_ERROR("Submatrices of a column-major panel matrix which are not horizontally aligned on a tile boundary "
                        "are currently not supported");
            }
            else
            {
                BLAZE_USER_ASSERT( row()    + rows()    <= matrix_.rows()   , "Invalid submatrix specification" );
                BLAZE_USER_ASSERT( column() + columns() <= matrix_.columns(), "Invalid submatrix specification" );
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
            BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
            BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

            return matrix_(row()+i,column()+j);
        }


        ConstReference operator()( size_t i, size_t j ) const
        {
            BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
            BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

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


        Pointer tile(size_t i, size_t j) noexcept
        {
            return data_ + spacing() * i + tileSize_ * tileSize_ * j;
        }
        

        ConstPointer tile(size_t i, size_t j) const noexcept
        {
            return data_ + spacing() * i + tileSize_ * tileSize_ * j;
        }


        Pointer ptr(size_t i, size_t j)
        {
            return matrix_.ptr(i + row(), j + column());
        }


        ConstPointer ptr(size_t i, size_t j) const
        {
            return matrix_.ptr(i + row(), j + column());
        }
        

    private:
        static size_t constexpr tileSize_ = TileSize_v<ElementType>;

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
        // BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE    ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE   ( MT );
        // BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( MT );
        BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( MT );
        BLAZEFEO_CONSTRAINT_MUST_NOT_BE_PANEL_SUBMATRIX_TYPE(MT);
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