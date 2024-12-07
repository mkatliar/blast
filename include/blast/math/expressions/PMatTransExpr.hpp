// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/constraints/PanelMatrix.hpp>
#include <blast/math/expressions/PanelMatrix.hpp>
#include <blast/math/TypeTraits.hpp>

#include <type_traits>


namespace blast
{
    //*************************************************************************************************
    /*!\brief Expression object for panel matrix transpositions.
    // \ingroup panel_matrix_expression
    //
    // The PMatTransExpr class represents the compile time expression for transpositions of
    // panel matrices.
    */
    template <typename MT, StorageOrder SO>
    class PMatTransExpr
    {
    public:
        using ElementType = ElementType_t<MT>;
        using Operand = MT const&;

        static StorageOrder constexpr storageOrder = SO;


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
        ElementType operator()(size_t i, size_t j) const
        {
            assert(i < pm_.columns());
            assert(j < pm_.rows());
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
        ElementType at(size_t i, size_t j) const
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


        /**
         * @brief Pointer to the data
         *
         * @return Pointer to matrix data
         */
        ElementType * data() noexcept
        {
            return pm_.data();
        }


        /**
         * @brief Constant pointer to the data
         *
         * @return Constant pointer to matrix data
         */
        ElementType const * data() const noexcept
        {
            return pm_.data();
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


    private:
        Operand pm_;
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
    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    inline decltype(auto) trans(MT const& pm)
    {
        using ReturnType = const PMatTransExpr<MT, !StorageOrder_v<MT>>;
        return ReturnType(pm);
    }


    template <typename MT, StorageOrder SO>
    struct IsPanelMatrix<PMatTransExpr<MT, SO>> : std::true_type {};


    template <typename MT, StorageOrder SO>
    struct IsStatic<PMatTransExpr<MT, SO>> : IsStatic<MT> {};


    template <typename MT, StorageOrder SO>
    struct IsAligned<PMatTransExpr<MT, SO>> : IsAligned<MT> {};


    template <typename MT, StorageOrder SO>
    struct IsPadded<PMatTransExpr<MT, SO>> : IsPadded<MT> {};


    /**
     * @brief Specialization for @a PMatTransExpr
     *
     * @tparam expression operand type
     * @tparam SO storage order
     */
    template <typename MT, StorageOrder SO>
    struct Spacing<PMatTransExpr<MT, SO>> : Spacing<MT> {};


    /**
     * @brief Specialization for @a PMatTransExpr
     *
     * @tparam expression operand type
     * @tparam SO storage order
     */
    template <typename MT, StorageOrder SO>
    struct StorageOrderHelper<PMatTransExpr<MT, SO>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};
}
