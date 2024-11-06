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
     * @brief Reference implementation of triangular matrix solver.
     *
     * Solves the matrix equation
     *
     * A * X = alpha * B
     *
     * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html/d9/de5/group__trsm_ga7120d931d7b1a15e12d50d328799df8a.html
     *
     * @tparam Real real number type
     * @tparam MPA matrix pointer type for the matrix @a A
     * @tparam MPX matrix pointer type for the matrix @a X
     * @tparam MPB matrix pointer type for the matrix @a B
     *
     * @param m the number of rows of @a B and @a X
     * @param n the number of columns of @a B and @a X
     * @param A pointer to a matrix of dimension ( @a m, @a m ). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param X pointer to a matrix of dimension ( @a m, @a n ) for the result. Can be equal to @a B.
     * @param alpha scalar multiplier
     * @param B pointer to a matrix of dimension ( @a m, @a n ).
     */
    template <typename Real, typename MPA, typename MPX, typename MPB>
    requires MatrixPointer<MPA, Real> && MatrixPointer<MPX, Real> && MatrixPointer<MPB, Real>
    inline void trsm(size_t m, size_t n, MPA A, UpLo uplo, bool diag, MPX X, Real alpha, MPB B)
    {
        for (size_t j = 0; j < n; ++j)
        {
            if (uplo == UpLo::Lower)
            {
                for (size_t i = 0; i < m; ++i)
                {
                    Real v = alpha * B[i, j];
                    for (size_t k = 0; k < i; ++k)
                        v -= A[i, k] * X[k, j];

                    X[i, j] = diag ? v : v / A[i, i];
                }
            }
            else // uplo == UpLo::Upper
            {
                for (size_t i = m; i-- > 0; )
                {
                    Real v = alpha * B[i, j];
                    for (size_t k = i + 1; k < m; ++k)
                        v -= A[i, k] * X[k, j];

                    X[i, j] = diag ? v : v / A[i, i];
                }
            }
        }
    }


    /**
     * @brief Reference implementation of triangular matrix solver.
     *
     * Solves the matrix equation
     *
     * X * A = alpha * B
     *
     * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html/d9/de5/group__trsm_ga7120d931d7b1a15e12d50d328799df8a.html
     *
     * @tparam Real real number type
     * @tparam MPX matrix pointer type for the matrix @a X
     * @tparam MPA matrix pointer type for the matrix @a A
     * @tparam MPB matrix pointer type for the matrix @a B
     *
     * @param m the number of rows of @a B and @a X
     * @param n the number of columns of @a B and @a X
     * @param A pointer to a matrix of dimension ( @a n, @a n ). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param X pointer to a matrix of dimension ( @a m, @a n ) for the result. Can be equal to @a B.
     * @param alpha scalar multiplier
     * @param B pointer to a matrix of dimension ( @a m, @a n ).
     */
    template <typename Real, typename MPX, typename MPA, typename MPB>
    requires MatrixPointer<MPA, Real> && MatrixPointer<MPX, Real> && MatrixPointer<MPB, Real>
    inline void trsm(size_t m, size_t n, MPX X, MPA A, UpLo uplo, bool diag, Real alpha, MPB B)
    {
        for (size_t i = 0; i < m; ++i)
        {
            if (uplo == UpLo::Upper)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    Real v = alpha * B[i, j];
                    for (size_t k = 0; k < j; ++k)
                        v -= X[i, k] * A[k, j];

                    X[i, j] = diag ? v : v / A[j, j];
                }
            }
            else // uplo == UpLo::Lower
            {
                for (size_t j = n; j-- > 0; )
                {
                    Real v = alpha * B[i, j];
                    for (size_t k = j + 1; k < n; ++k)
                        v -= X[i, k] * A[k, j];

                    X[i, j] = diag ? v : v / A[j, j];
                }
            }
        }
    }


    /**
     * @brief Reference implementation of triangular matrix solver.
     *
     * Solves the matrix equation
     *
     * A * X = alpha * B
     *
     * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html/d9/de5/group__trsm_ga7120d931d7b1a15e12d50d328799df8a.html
     *
     * @tparam Real real number type
     * @tparam MTA type for the matrix @a A
     * @tparam MTX type for the matrix @a X
     * @tparam MTB type for the matrix @a B
     *
     * @param A matrix of dimension (m, m). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param X matrix of dimension (m, n) for the result. Can be equal to @a B.
     * @param alpha scalar multiplier
     * @param B matrix of dimension (m, n).
     */
    template <typename Real, typename MTA, typename MTX, typename MTB>
    requires Matrix<MTA, Real> && Matrix<MTX, Real> && Matrix<MTB, Real>
    inline void trsm(MTA const& A, UpLo uplo, bool diag, MTX& X, Real alpha, MTB const& B)
    {
        size_t const m = rows(B);
        size_t const n = columns(B);

        if (rows(A) != m || columns(A) != m ||
            rows(X) != m || columns(X) != n)
            throw std::invalid_argument {"Inconsistent matrix sizes"};

        reference::trsm(m, n, ptr(A), uplo, diag, ptr(X), alpha, ptr(B));
    }


    /**
     * @brief Reference implementation of triangular matrix solver.
     *
     * Solves the matrix equation
     *
     * X * A = alpha * B
     *
     * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     * non-unit, upper or lower triangular matrix.
     *
     * LAPACK reference: https://netlib.org/lapack/explore-html/d9/de5/group__trsm_ga7120d931d7b1a15e12d50d328799df8a.html
     *
     * @tparam Real real number type
     * @tparam MTA type for the matrix @a A
     * @tparam MTX type for the matrix @a X
     * @tparam MTB type for the matrix @a B
     *
     * @param X matrix of dimension (m, n) for the result. Can be equal to @a B.
     * @param A matrix of dimension (n, n). Depending on the value of @a uplo, the
     *     upper (lower) triangular part of @a A must contain the upper (lower) triangular matrix
     *     and the strictly lower (upper) triangular part of @a A is not referenced. When @a diag == true, the diagonal elements of
     *     @a A are not referenced either, but are assumed to be unity.
     * @param uplo specifies whether the matrix @a A is an upper or lower triangular matrix
     * @param diag specifies whether or not @a A is unit triangular
     * @param alpha scalar multiplier
     * @param B matrix of dimension (m, n).
     */
    template <typename Real, typename MTX, typename MTA, typename MTB>
    requires Matrix<MTX, Real> && Matrix<MTA, Real> && Matrix<MTB, Real>
    inline void trsm(MTX& X, MTA const& A, UpLo uplo, bool diag, Real alpha, MTB const& B)
    {
        size_t const m = rows(B);
        size_t const n = columns(B);

        if (rows(A) != n || columns(A) != n ||
            rows(X) != m || columns(X) != n)
            throw std::invalid_argument {"Inconsistent matrix sizes"};

        reference::trsm(m, n, ptr(X), ptr(A), uplo, diag, alpha, ptr(B));
    }
}
