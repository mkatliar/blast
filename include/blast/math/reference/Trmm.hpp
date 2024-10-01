// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/UpLo.hpp>
#include <blast/util/Types.hpp>


namespace blast :: reference
{
    /**
     * @brief Reference implementation of left triangular matrix multiplication.
     *
     * Performs the matrix-matrix operation
     *
     * C := alpha * A * B
     *
     * where alpha is a scalar, B is an m by n matrix, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
     *
     * @tparam Real real number type
     * @tparam MPA matrix pointer type for the matrix @a A
     * @tparam MPB matrix pointer type for the matrix @a B
     * @tparam MPC matrix pointer type for the matrix @a C
     *
     * @param m number of rows in @a B and @a C
     * @param n number of columns in @a B and @a C
     * @param alpha scalar multiplier
     * @param A pointer to a matrix of dimension ( @a m, @a m ). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param B pointer to a matrix of dimension ( @a m, @a n ).
     * @param C pointer to a matrix of dimension ( @a m, @a n ) for the result. Can be equal to @a B.
     */
    template <typename Real, typename MPA, typename MPB, typename MPC>
    requires MatrixPointer<MPA, Real> && MatrixPointer<MPB, Real> && MatrixPointer<MPC, Real>
    inline void trmm(size_t m, size_t n, Real alpha, MPA A, UpLo uplo, bool diag, MPB B, MPC C)
    {
        for (size_t j = 0; j < n; ++j)
        {
            if (uplo == UpLo::Upper)
            {
                for (size_t i = 0; i < m; ++i)
                {
                    Real v {};
                    for (size_t k = i; k < m; ++k)
                        v += (diag && k == i) ? *(~B)(k, j) : *(~A)(i, k) * *(~B)(k, j);

                    *(~C)(i, j) = alpha * v;
                }
            }
            else
            {
                for (size_t i = m; i-- > 0; )
                {
                    Real v {};
                    for (size_t k = 0; k <= i; ++k)
                        v += (diag && k == i) ? *(~B)(k, j) : *(~A)(i, k) * *(~B)(k, j);

                    *(~C)(i, j) = alpha * v;
                }
            }
        }
    }


    /**
     * @brief Reference implementation of right triangular matrix multiplication.
     *
     * Performs the matrix-matrix operation
     *
     * C := alpha * B * A
     *
     * where alpha is a scalar, B is an m by n matrix, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
     *
     * @tparam Real real number type
     * @tparam MPB matrix pointer type for the matrix @a B
     * @tparam MPA matrix pointer type for the matrix @a A
     * @tparam MPC matrix pointer type for the matrix @a C
     *
     * @param m number of rows in @a B and @a C
     * @param n number of columns in @a B and @a C
     * @param alpha scalar multiplier
     * @param B pointer to a matrix of dimension ( @a m, @a n ).
     * @param A pointer to a matrix of dimension ( @a n, @a n ). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param C pointer to a matrix of dimension ( @a m, @a n ) for the result. Can be equal to @a B.
     */
    template <typename Real, typename MPB, typename MPA, typename MPC>
    requires MatrixPointer<MPA, Real> && MatrixPointer<MPB, Real> && MatrixPointer<MPC, Real>
    inline void trmm(size_t m, size_t n, Real alpha, MPB B, MPA A, UpLo uplo, bool diag, MPC C)
    {
        for (size_t i = 0; i < m; ++i)
        {
            if (uplo == UpLo::Lower)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    Real v {};
                    for (size_t k = j; k < n; ++k)
                        v += (diag && k == j) ? *(~B)(i, k) : *(~B)(i, k) * *(~A)(k, j);

                    *(~C)(i, j) = alpha * v;
                }
            }
            else
            {
                for (size_t j = n; j-- > 0; )
                {
                    Real v {};
                    for (size_t k = 0; k <= j; ++k)
                        v += (diag && k == j) ? *(~B)(i, k) : *(~B)(i, k) * *(~A)(k, j);

                    *(~C)(i, j) = alpha * v;
                }
            }
        }
    }


    /**
     * @brief Reference implementation of left triangular matrix multiplication.
     *
     * Performs the matrix-matrix operation
     *
     * C := alpha * A * B
     *
     * where alpha is a scalar, B is an m by n matrix, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
     *
     * @tparam Real real number type
     * @tparam MTA matrix type for the matrix @a A
     * @tparam MTB matrix type for the matrix @a B
     * @tparam MTC matrix type for the matrix @a C
     *
     * @param alpha scalar multiplier
     * @param A a matrix of dimension (m, m). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param B a matrix of dimension (m, n).
     * @param C a matrix of dimension (m, n) for the result. Can be the same matrix as @a B.
     *
     * @throw @a std::invalid_argument if matrix sizes are inconsistent
     */
    template <typename Real, typename MTA, typename MTB, typename MTC>
    requires Matrix<MTA, Real> && Matrix<MTB, Real> && Matrix<MTC, Real>
    inline void trmm(Real alpha, MTA const& A, UpLo uplo, bool diag, MTB const& B, MTC& C)
    {
        size_t const m = rows(B);
        size_t const n = columns(B);

        if (rows(A) != m || columns(A) != m ||
            rows(C) != m || columns(C) != n)
            throw std::invalid_argument {"Inconsistent matrix sizes"};

        trmm(m, n, alpha, ptr(A), uplo, diag, ptr(B), ptr(C));
    }


    /**
     * @brief Reference implementation of right triangular matrix multiplication.
     *
     * Performs the matrix-matrix operation
     *
     * C := alpha * B * A
     *
     * where alpha is a scalar, B is an m by n matrix, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
     *
     * @tparam Real real number type
     * @tparam MTB matrix type for the matrix @a B
     * @tparam MTA matrix type for the matrix @a A
     * @tparam MTC matrix type for the matrix @a C
     *
     * @param alpha scalar multiplier
     * @param B a matrix of dimension (m, n).
     * @param A a matrix of dimension (m, m). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param C a matrix of dimension (m, n) for the result. Can be the same matrix as @a B.
     *
     * @throw @a std::invalid_argument if matrix sizes are inconsistent
     */
    template <typename Real, typename MTB, typename MTA, typename MTC>
    requires Matrix<MTA, Real> && Matrix<MTB, Real> && Matrix<MTC, Real>
    inline void trmm(Real alpha, MTB const& B, MTA const& A, UpLo uplo, bool diag, MTC& C)
    {
        size_t const m = rows(B);
        size_t const n = columns(B);

        if (rows(A) != n || columns(A) != n ||
            rows(C) != m || columns(C) != n)
            throw std::invalid_argument {"Inconsistent matrix sizes"};

        trmm(m, n, alpha, ptr(B), ptr(A), uplo, diag, ptr(C));
    }
}
