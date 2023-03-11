// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/Forward.hpp>
#include <blazefeo/math/panel/Gemm.hpp>
#include <blazefeo/math/views/submatrix/BaseTemplate.hpp>
#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/panel/PanelSize.hpp>
#include <blazefeo/system/CacheLine.hpp>

#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/util/constraints/Vectorizable.h>

#include <initializer_list>


namespace blazefeo
{
    using namespace blaze;


    /// @brief Panel matrix with statically defined size.
    ///
    /// @tparam Type element type of the matrix
    /// @tparam M number of rows
    /// @tparam N number of columns
    /// @tparam SO storage order of panel elements
    template <typename Type, size_t M, size_t N, bool SO = columnMajor>
    class StaticPanelMatrix
    :   public PanelMatrix<StaticPanelMatrix<Type, M, N, SO>, SO>
    {
    public:
        //**Type definitions****************************************************************************
        using This          = StaticPanelMatrix<Type, M, N, SO>;   //!< Type of this StaticPanelMatrix instance.
        using BaseType      = PanelMatrix<This, SO>;        //!< Base type of this StaticPanelMatrix instance.
        using ResultType    = This;                        //!< Result type for expression template evaluations.
        using OppositeType  = StaticPanelMatrix<Type, M, N, !SO>;  //!< Result type with opposite storage order for expression template evaluations.
        using TransposeType = StaticPanelMatrix<Type, N, M, !SO>;  //!< Transpose type for expression template evaluations.
        using ElementType   = Type;                        //!< Type of the matrix elements.
        // using SIMDType      = SIMDTrait_t<ElementType>;    //!< SIMD type of the matrix elements.
        using ReturnType    = const Type&;                 //!< Return type for expression template evaluations.
        using CompositeType = const This&;                 //!< Data type for composite expression templates.

        using Reference      = Type&;        //!< Reference to a non-constant matrix value.
        using ConstReference = const Type&;  //!< Reference to a constant matrix value.
        using Pointer        = Type*;        //!< Pointer to a non-constant matrix value.
        using ConstPointer   = const Type*;  //!< Pointer to a constant matrix value.

        using Iterator      = DenseIterator<Type, aligned>;        //!< Iterator over non-constant elements.
        using ConstIterator = DenseIterator<const Type, aligned>;  //!< Iterator over constant elements.


        StaticPanelMatrix()
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_, capacity_, Type {});
        }


        constexpr StaticPanelMatrix(std::initializer_list<std::initializer_list<Type>> list)
        {
            std::fill_n(v_, capacity_, Type {});

            if (list.size() != M || determineColumns(list) > N)
                BLAZE_THROW_INVALID_ARGUMENT("Invalid setup of static panel matrix");

            size_t i = 0;

            for (auto const& row : list)
            {
                size_t j = 0;

                for (const auto& element : row)
                {
                    v_[elementIndex(i, j)] = element;
                    ++j;
                }

                ++i;
            }
        }


        StaticPanelMatrix& operator=(Type val)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        template< typename MT    // Type of the right-hand side matrix
                , bool SO2 >      // Storage order of the right-hand side matrix
        StaticPanelMatrix& operator=(blaze::Matrix<MT, SO2> const& rhs);


        constexpr ConstReference operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        constexpr Reference operator()(size_t i, size_t j)
        {
            return v_[elementIndex(i, j)];
        }


        size_t constexpr rows() const
        {
            return M;
        }


        size_t constexpr columns() const
        {
            return N;
        }


        size_t constexpr spacing() const
        {
            return spacing_;
        }


        size_t constexpr panels() const
        {
            return panels_;
        }


        /// @brief Offset of the first matrix element from the start of the panel.
        ///
        /// In rows for column-major matrices, in columns for row-major matrices.
        size_t constexpr offset() const
        {
            return 0;
        }


        Type * data() noexcept
        {
            return v_;
        }


        Type const * data() const noexcept
        {
            return v_;
        }


        Type * ptr(size_t i, size_t j)
        {
            // BLAZE_USER_ASSERT(i % panelSize_ == 0, "Row index not aligned to panel boundary");
            return v_ + elementIndex(i, j);
        }


        Type const * ptr(size_t i, size_t j) const
        {
            // BLAZE_USER_ASSERT(i % panelSize_ == 0, "Row index not aligned to panel boundary");
            return v_ + elementIndex(i, j);
        }


        template <size_t SS>
        auto load(size_t i, size_t j) const
        {
            BLAZE_INTERNAL_ASSERT(i < M, "Invalid row access index");
            BLAZE_INTERNAL_ASSERT(j < N, "Invalid column access index");
            BLAZE_INTERNAL_ASSERT(i % panelSize_ == 0 || SO == rowMajor, "Row index not aligned to panel boundary");
            BLAZE_INTERNAL_ASSERT(j % panelSize_ == 0 || SO == columnMajor, "Column index not aligned to panel boundary");

            return blazefeo::load<aligned, SS>(v_ + elementIndex(i, j));
        }


        template <typename T>
        void store(size_t i, size_t j, T val)
        {
            BLAZE_INTERNAL_ASSERT(i < M, "Invalid row access index");
            BLAZE_INTERNAL_ASSERT(j < N, "Invalid column access index");
            BLAZE_INTERNAL_ASSERT(i % panelSize_ == 0 || SO == rowMajor, "Row index not aligned to panel boundary");
            BLAZE_INTERNAL_ASSERT(j % panelSize_ == 0 || SO == columnMajor, "Column index not aligned to panel boundary");

            // We never use maskstore here because we have padding
            blazefeo::store<aligned>(v_ + elementIndex(i, j), val);
        }


    private:
        static size_t constexpr panelSize_ = PanelSize_v<Type>;
        static size_t constexpr elementsPerTile_ = panelSize_ * panelSize_;
        static size_t constexpr tileRows_ = M / panelSize_ + (M % panelSize_ > 0);
        static size_t constexpr tileColumns_ = N / panelSize_ + (N % panelSize_ > 0);
        static size_t constexpr panels_ = SO == columnMajor ? tileRows_ : tileColumns_;
        static size_t constexpr spacing_ = (SO == columnMajor ? tileColumns_ : tileRows_) * elementsPerTile_;
        static size_t constexpr capacity_ = panels_ * spacing_;

        // Alignment of the data elements.
        static size_t constexpr alignment_ = CACHE_LINE_SIZE;

        // Aligned element storage.
        alignas(alignment_) Type v_[capacity_];


        size_t elementIndex(size_t i, size_t j) const
        {
            return SO == columnMajor
                ? i / panelSize_ * spacing_ + i % panelSize_ + j * panelSize_
                : j / panelSize_ * spacing_ + j % panelSize_ + i * panelSize_;
        }


        BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);
    };


    template <typename Type, size_t M, size_t N, bool SO>
    template< typename MT    // Type of the right-hand side matrix
            , bool SO2 >      // Storage order of the right-hand side matrix
    inline StaticPanelMatrix<Type, M, N, SO>& StaticPanelMatrix<Type, M, N, SO>::operator=(blaze::Matrix<MT, SO2> const& rhs)
    {
        // using blaze::assign;

        // using TT = decltype( trans( *this ) );
        // using CT = decltype( ctrans( *this ) );
        // using IT = decltype( inv( *this ) );

        // if( (*rhs).rows() != M || (*rhs).columns() != N ) {
        //     BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to static matrix" );
        // }

        // if( IsSame_v<MT,TT> && (*rhs).isAliased( this ) ) {
        //     transpose( typename IsSquare<This>::Type() );
        // }
        // else if( IsSame_v<MT,CT> && (*rhs).isAliased( this ) ) {
        //     ctranspose( typename IsSquare<This>::Type() );
        // }
        // else if( !IsSame_v<MT,IT> && (*rhs).canAlias( this ) ) {
        //     StaticPanelMatrix tmp( *rhs );
        //     assign( *this, tmp );
        // }
        // else {
        //     if( IsSparseMatrix_v<MT> )
        //         reset();
        //     assign( *this, *rhs );
        // }

        // BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

        assign(*this, *rhs);

        return *this;
    }
}

namespace blaze
{
    //=================================================================================================
    //
    //  HASMUTABLEDATAACCESS SPECIALIZATIONS
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    template <typename T, size_t M, size_t N, bool SO>
    struct HasMutableDataAccess<blazefeo::StaticPanelMatrix<T, M, N, SO>>
    :   public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************
}