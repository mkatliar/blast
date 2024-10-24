// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/util/Types.hpp>

#include <type_traits>
#include <cassert>


namespace blast
{
    /**
     * @brief Expression object for matrix transpositions
     *
     * @tparam MT type of the matrix being transposed
     *
     */
    template <Matrix MT>
    class MatTransExpr
    {
    public:
        using ElementType = ElementType_t<MT>;
        using Operand = MT const;

        explicit MatTransExpr(MT const& matrix) noexcept
        :   m_ {matrix}
        {}


        /**
         * @brief 2D-access to the matrix elements.
         *
         * @param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
         * @param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
         *
         * @return The resulting value.
         */
        ElementType operator()(size_t i, size_t j) const noexcept
        {
            assert(i < columns(m_));
            assert(j < rows(m_));
            return m_(j, i);
        }


        /**
         * @brief Number of rows
         *
         * @param e matrix transpose expression
         *
         * @return The number of rows of the transpose expression.
         */
        friend size_t rows(MatTransExpr const& e) noexcept
        {
            return columns(e.m_);
        }


        /**
         * @brief Number of columns
         *
         * @param e matrix transpose expression
         *
         * @return The number of columns of the transpose expression.
         */
        friend size_t columns(MatTransExpr const& e) noexcept
        {
            return rows(e.m_);
        }


        /**
         * @brief Pointer to the data
         *
         * @param e matrix transpose expression
         *
         * @return Pointer to matrix data
         */
        friend ElementType * data(MatTransExpr& e) noexcept
        {
            return data(e.m_);
        }


        /**
         * @brief Constant pointer to the data
         *
         * @param e matrix transpose expression
         *
         * @return Constant pointer to matrix data
         */
        friend ElementType const * data(MatTransExpr const& e) noexcept
        {
            return data(e.m_);
        }


        /**
         * @brief Returns the spacing of the underlying matrix storage.
         *
         * @param e matrix transpose expression
         *
         * @return The spacing of the underlying matrix storage.
         */
        friend size_t spacing(MatTransExpr const& e) noexcept
        {
            return spacing(e.m_);
        }


        /**
         * @brief Returns the matrix operand.
         *
         * @return The matrix operand.
         */
        Operand const& operand() const noexcept
        {
            return m_;
        }

    private:
        Operand& m_;
    };


    /**
     * @brief Transpose of a matrix
     *
     * @tparam MT matrix type
     *
     * @param m matrix
     *
     * @return expression representing the transposition of the matrix
     */
    template <Matrix MT>
    inline auto trans(MT const& m)
    {
        return MatTransExpr<MT> {m};
    }


    /**
     * @brief Transpose of a transpose
     *
     * @tparam MT matrix type
     *
     * @param e matrix transposition expression
     *
     * @return the operand of the transposition expression
     */
    template <typename MT>
    inline decltype(auto) trans(MatTransExpr<MT> const& e)
    {
        return e.operand();
    }


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct IsAligned<MatTransExpr<MT>> : IsAligned<MT> {};


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct IsPadded<MatTransExpr<MT>> : IsPadded<MT> {};


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct IsStatic<MatTransExpr<MT>> : IsStatic<MT> {};


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct IsDenseMatrix<MatTransExpr<MT>> : IsDenseMatrix<MT> {};


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct IsPanelMatrix<MatTransExpr<MT>> : IsPanelMatrix<MT> {};


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct Spacing<MatTransExpr<MT>> : Spacing<MT> {};


    /**
     * @brief Specialization for @a MatTransExpr
     *
     * @tparam MT expression operand type
     */
    template <typename MT>
    struct StorageOrderHelper<MatTransExpr<MT>> : std::integral_constant<StorageOrder, !StorageOrder_v<MT>> {};
}
