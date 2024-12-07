// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/views/row/BaseTemplate.hpp>
#include <blast/math/views/row/Panel.hpp>
#include <blast/math/TypeTraits.hpp>


namespace blast
{
    /**
     * @brief Row view of a matrix
     *
     * NOTE: this implementation is not optimized!
     * It holds a refrerence to the matrix, as well as a raw pointer,
     * the row index, and the column index of the first element.
     * Instead it could be just a matrix pointer and the length.
     * This would reduce the amount of data needed to represent the @a Row object,
     * increasig the possibility of storing everything in registers and reducing
     * the number of registers needed.
     *
     * @tparam MT viewed matrix type
     */
    template <Matrix MT>
    class Row
    {
    public:
        static TransposeFlag constexpr transposeFlag = rowVector;

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
         * @param matrix the matrix
         * @param i row index
         * @param j start of the row
         * @param n row length
         */
        explicit inline constexpr Row(MT& matrix, size_t i, size_t j, size_t n)
        :   matrix_ {matrix}
        ,   i_ {i}
        ,   j_ {j}
        ,   n_ {n}
        ,   data_ {&matrix(i, j)}
        {
            BLAST_USER_ASSERT(i_ < rows(matrix_), "Invalid row index");
            BLAST_USER_ASSERT(j_ < columns(matrix_), "Invalid column index");
            BLAST_USER_ASSERT(j_ + n_ <= columns(matrix_), "Invalid row length");
        }


        Row(Row const&) = default;


        /**
         * @brief Vector assignment
         *
         * Copies elements from the right-hand side expression
         *
         * @tparam VT type of right-hand side vector
         *
         * @param rhs
         *
         * @return reference to *this
         */
        template <Vector VT>
        Row& operator=(VT const& rhs)
        {
            assign(*this, rhs);
            return *this;
        }


        Reference operator[](size_t j) noexcept
        {
            BLAST_USER_ASSERT(j_ + j < columns(matrix_), "Invalid vector access index");

            return matrix_(i_, j_ + j);
        }


        ConstReference operator[](size_t j) const
        {
            BLAST_USER_ASSERT(j_ + j < columns(matrix_), "Invalid vector access index");

            return const_cast<MT const&>(matrix_)(i_, j_ + j);
        }


        friend Pointer data(Row& m) noexcept
        {
            return m.data_;
        }


        friend ConstPointer data(Row const& m) noexcept
        {
            return m.data_;
        }


        friend size_t constexpr size(Row const& row) noexcept
        {
            return row.n_;
        }


        /**
         * @brief Subvector of a row
         *
         * @param row the original row
         * @param j start index of the subvector within @a row
         * @param n length of the subvector
         */
        friend auto subvector(Row&& row, size_t j, size_t n) noexcept
        {
            return Row<MT> {row.matrix_, row.i_, row.j_ + j, n};
        }


        /**
         * @brief Subvector of a const row
         *
         * @param row the original row
         * @param j start index of the subvector within @a row
         * @param n length of the subvector
         */
        friend auto subvector(Row const& row, size_t j, size_t n) noexcept
        {
            return Row<MT const> {const_cast<MT const&>(row.matrix_), row.i_, row.j_ + j, n};
        }


    private:
        ViewedType& matrix_;        //!< The matrix containing the submatrix.
        size_t const i_;
        size_t const j_;
        size_t const n_;
        Pointer const data_;
    };


    /**
     * @brief Specialization for @a Row class
     */
    template <typename MT>
    struct IsAligned<Row<MT>> : std::integral_constant<bool, IsAligned_v<MT> && IsPadded_v<MT>> {};


    /**
     * @brief Specialization for @a Row class
     */
    template <typename MT>
    struct IsPadded<Row<MT>> : std::integral_constant<bool, IsPadded_v<MT> && StorageOrder_v<MT> == rowMajor> {};


    /**
     * @brief Specialization for @a Row class
     */
    template <typename MT>
    struct IsStatic<Row<MT>> : IsStatic<MT> {};


    /**
     * @brief Specialization for @a Row class
     */
    template <typename MT>
    struct IsDenseVector<Row<MT>> : IsDenseMatrix<MT> {};


    /**
     * @brief Specialization for @a Row class
     */
    template <typename MT>
    struct IsView<Row<MT>> : std::integral_constant<bool, true> {};


    /**
     * @brief Specialization for rows of dense (non-panel) matrices
     */
    template <typename MT>
    requires IsDenseMatrix_v<MT>
    struct IsStaticallySpaced<Row<MT>> : std::integral_constant<bool, IsStatic_v<MT> || StorageOrder_v<MT> == rowMajor> {};


    /**
     * @brief Specialization for rows of panel matrices
     */
    template <typename MT>
    requires IsPanelMatrix_v<MT>
    struct IsStaticallySpaced<Row<MT>> : std::integral_constant<bool, StorageOrder_v<MT> == columnMajor> {};


    /**
     * @brief Specialization for rows of dense (non-panel) matrices
     */
    template <typename MT>
    requires IsDenseMatrix_v<MT> && IsStatic_v<MT>
    struct Spacing<Row<MT>> : std::integral_constant<size_t, StorageOrder_v<MT> == rowMajor ? 1 : Spacing_v<MT>> {};


    /**
     * @brief Specialization for rows of panel matrices
     */
    template <typename MT>
    requires IsPanelMatrix_v<MT> && (StorageOrder_v<MT> == columnMajor)
    struct Spacing<Row<MT>> : std::integral_constant<size_t, SimdSize_v<ElementType_t<MT>>> {};


    /**
     * @brief Full row of a matrix
     *
     * @tparam MT type of the matrix containing the row
     *
     * @param matrix matrix containing the row
     * @param row row index
     *
     * @return submatrix of @a matrix
     */
    template <typename MT>
    inline auto row(MT& matrix, size_t row)
    {
        return Row<MT> {matrix, row, 0, columns(matrix)};
    }


    /**
     * @brief Partial row of a matrix
     *
     * @tparam MT type of the matrix containing the row
     *
     * @param matrix matrix containing the row
     * @param row row index
     * @param column row start
     * @param n row length
     *
     * @return submatrix of @a matrix
     */
    template <typename MT>
    inline auto row(MT& matrix, size_t row, size_t column, size_t n)
    {
        return Row<MT> {matrix, row, column, n};
    }


    /**
     * @brief Set all elements of a row to their default value (0).
     *
     * @param matrix submatrix to set to 0.
     */
    template <typename MT>
    inline void reset(Row<MT>& row) noexcept
    {
        for (size_t i = 0; i < size(row); ++i)
            row[i] = 0;
    }
}
