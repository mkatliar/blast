// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/views/submatrix/BaseTemplate.hpp>
#include <blast/math/views/submatrix/Panel.hpp>
#include <blast/math/TypeTraits.hpp>

#include <type_traits>


namespace blast
{
    /**
     * @brief Submatrix view of a matrix
     *
     * @tparam MT viewed matrix type
     * @tparam AF whether the submatrix lop left element is aligned
     */
    template <Matrix MT, AlignmentFlag AF>
    class Submatrix
    {
    public:
        using ViewedType = MT;
        using ElementType = ElementType_t<MT>;

        //! Reference to a constant submatrix value.
        using ConstReference = ElementType const&;

        //! Reference to a non-constant submatrix value.
        using Reference = std::conditional_t<IsConst_v<MT>, ConstReference, ElementType&>;

        //! Pointer to a constant submatrix value.
        using ConstPointer = ElementType const *;

        //! Pointer to a non-constant submatrix value.
        using Pointer = std::conditional_t<IsConst_v<MT>, ConstPointer, ElementType *>;


        /**
         * @brief Constructor
         *
         * @param i top row index of the submatrix
         * @param j left column index of the submatrix
         * @param m number of rows of the submatrix
         * @param n number of columns of the submatrix
         */
        explicit inline constexpr Submatrix(MT& matrix, size_t i, size_t j, size_t m, size_t n)
        :   matrix_ {matrix}
        ,   i_ {i}
        ,   j_ {j}
        ,   m_ {m}
        ,   n_ {n}
        ,   data_ {&matrix_(i_, j_)}
        {
            BLAST_USER_ASSERT(i_ + m_ <= rows(matrix_), "Invalid submatrix specification");
            BLAST_USER_ASSERT(j_ + n_ <= columns(matrix_), "Invalid submatrix specification");
        }


        Submatrix(Submatrix const&) = default;


        /**
         * @brief Matrix assignment
         *
         * Copies elements from the right-hand side expression
         *
         * @tparam MT2 type of right-hand side matrix
         *
         * @param rhs
         *
         * @return reference to *this
         */
        template <Matrix MT2>
        Submatrix& operator=(MT2 const& rhs)
        {
            assign(*this, rhs);
            return *this;
        }


        size_t constexpr row() const noexcept
        {
            return i_;
        };


        size_t constexpr column() const noexcept
        {
            return j_;
        };


        friend size_t constexpr rows(Submatrix const& m) noexcept
        {
            return m.m_;
        };


        friend size_t constexpr columns(Submatrix const& m) noexcept
        {
            return m.n_;
        };


        MT& operand() noexcept
        {
            return matrix_;
        }


        const MT& operand() const noexcept
        {
            return matrix_;
        }


        friend size_t spacing(Submatrix const& m) noexcept
        {
            return spacing(m.matrix_);
        }


        Reference operator()(size_t i, size_t j) noexcept
        {
            BLAST_USER_ASSERT(i < m_, "Invalid row access index");
            BLAST_USER_ASSERT(j < n_, "Invalid column access index");

            return matrix_(i_ + i, j_ + j);
        }


        ConstReference operator()( size_t i, size_t j ) const
        {
            BLAST_USER_ASSERT(i < m_, "Invalid row access index");
            BLAST_USER_ASSERT(j < n_, "Invalid column access index");

            return const_cast<MT const&>(matrix_)(i_ + i, j_ + j);
        }


        friend Pointer data(Submatrix& m) noexcept
        {
            return m.data_;
        }


        friend ConstPointer data(Submatrix const& m) noexcept
        {
            return m.data_;
        }


    private:
        ViewedType& matrix_;        //!< The matrix containing the submatrix.
        size_t const i_;
        size_t const j_;
        size_t const m_;
        size_t const n_;

        // Pointer to the first element of the submatrix
        Pointer const data_;
    };


    /**
     * @brief Specialization for @a Submatrix class
     */
    template <typename MT, AlignmentFlag AF>
    struct IsAligned<Submatrix<MT, AF>> : std::integral_constant<bool, AF> {};


    /**
     * @brief Specialization for @a Submatrix class
     */
    template <typename MT, AlignmentFlag AF>
    struct IsStatic<Submatrix<MT, AF>> : IsStatic<MT> {};


    /**
     * @brief Specialization for @a Submatrix class
     */
    template <typename MT, AlignmentFlag AF>
    struct IsDenseMatrix<Submatrix<MT, AF>> : IsDenseMatrix<MT> {};


    /**
     * @brief Specialization for @a Submatrix class
     */
    template <typename MT, AlignmentFlag AF>
    struct IsView<Submatrix<MT, AF>> : std::integral_constant<bool, true> {};


    /**
     * @brief Specialization for @a Submatrix class
     */
    template <typename MT, AlignmentFlag AF>
    struct Spacing<Submatrix<MT, AF>> : Spacing<MT> {};


    /**
     * @brief Specialization for @a Submatrix class
     */
    template <typename MT, AlignmentFlag AF>
    struct StorageOrderHelper<Submatrix<MT, AF>> : StorageOrderHelper<MT> {};


    /**
     * @brief Submatrix of a matrix
     *
     * @tparam AF alignment flag of the resulting submatrix
     * @tparam MT type of the matrix containing the submatrix
     *
     * @param matrix matrix containing the submatrix
     * @param row top row index of the submatrix
     * @param column left column index of the submatrix
     * @param m number of rows in the resulting submatrix
     * @param n number of columns in the resulting submatrix
     *
     * @return submatrix of @a matrix
     */
    template <AlignmentFlag AF, typename MT>
    inline decltype(auto) submatrix(MT& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return Submatrix<MT, AF> {matrix, row, column, m, n};
    }


    /**
     * @brief Submatrix of a const @a Submatrix
     *
     * @tparam AF alignment flag of the resulting submatrix
     * @tparam MT type of the matrix containing the submatrix
     * @tparam AF1 alignment flag of the matrix containing the submatrix
     *
     * @param matrix matrix containing the submatrix
     * @param row top row index of the submatrix (relative to @a matrix)
     * @param column left column index of the submatrix (relative to @a matrix)
     * @param m number of rows in the resulting submatrix
     * @param n number of columns in the resulting submatrix
     *
     * @return submatrix of @a matrix
     */
    template <AlignmentFlag AF, typename MT, AlignmentFlag AF1>
    inline decltype(auto) submatrix(Submatrix<MT, AF1> const& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return Submatrix<MT const, AF> {matrix.operand(), matrix.row() + row, matrix.column() + column, m, n};
    }


    /**
     * @brief Submatrix of a @a Submatrix
     *
     * @tparam AF alignment flag of the resulting submatrix
     * @tparam MT type of the matrix containing the submatrix
     * @tparam AF1 alignment flag of the matrix containing the submatrix
     *
     * @param matrix matrix containing the submatrix
     * @param row top row index of the submatrix (relative to @a matrix)
     * @param column left column index of the submatrix (relative to @a matrix)
     * @param m number of rows in the resulting submatrix
     * @param n number of columns in the resulting submatrix
     *
     * @return submatrix of @a matrix
     */
    template <AlignmentFlag AF, typename MT, AlignmentFlag AF1>
    inline decltype(auto) submatrix(Submatrix<MT, AF1>& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return Submatrix<MT, AF> {matrix.operand(), matrix.row() + row, matrix.column() + column, m, n};
    }


    /**
     * @brief Set all elements of a submatrix to their default value (0).
     *
     * @param matrix submatrix to set to 0.
     */
    template <typename MT, AlignmentFlag AF>
    inline void reset(Submatrix<MT, AF>& matrix) noexcept
    {
        for (size_t i = 0; i < rows(matrix); ++i)
            for (size_t j = 0; j < columns(matrix); ++j)
                matrix(i, j) = ElementType_t<MT> {};
    }
}
