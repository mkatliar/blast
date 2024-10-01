// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/algorithm/Tile.hpp>
#include <blast/math/register_matrix/Gemm.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/util/Exception.hpp>


namespace blast :: reference
{
    /**
     * @brief Constant times matrix plus matrix, matrix pointer arguments
     *
     * C := alpha*A + B
     *
     * where alpha is a scalar, and A, B and C are M by N matrices
     *
     * @tparam ST scalar type for @a alpha
     * @tparam MPA matrix pointer type for @a A
     * @tparam MPB matrix pointer type for @a B
     * @tparam MPC matrix pointer type for @a C
     *
     * @param M the number of rows of the matrices A, B, and C.
     * @param N the number of columns of the matrices A, B and C.
     * @param alpha the scalar alpha
     * @param A pointer to the top left element of matrix A
     * @param B pointer to the top left element of matrix B
     * @param C pointer to the top left element of matrix C
     */
    template <typename ST, MatrixPointer MPA, MatrixPointer MPB, MatrixPointer MPC>
    inline void axpy(size_t M, size_t N, ST alpha, MPA A, MPB B, MPC C)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                *(~C)(i, j) = alpha * *(~A)(i, j) + *(~B)(i, j);
    }


    /**
     * @brief Constant times matrix plus matrix, matrix arguments
     *
     * C := alpha*A + B
     *
     * where alpha is a scalar, and A, B and C are M by N matrices
     *
     * @tparam ST scalar type for @a alpha
     * @tparam MTA matrix type for @a A
     * @tparam MTB matrix type for @a B
     * @tparam MTC matrix type for @a C
     *
     * @param alpha the scalar alpha
     * @param A pointer to the top left element of matrix A
     * @param B pointer to the top left element of matrix B
     * @param C pointer to the top left element of matrix C
     *
     * @throw @a std::invalid_argument if matrix sizes are not consistent
     */
    template <typename ST, Matrix MTA, Matrix MTB, Matrix MTC>
    inline void axpy(ST alpha, MTA const& A, MTB const& B, MTC& C)
    {
        size_t const M = rows(C);
        size_t const N = columns(C);

        if (rows(A) != M || columns(A) != N ||
            rows(B) != M || columns(B) != N)
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Inconsistent matrix sizes"});

        reference::axpy(M, N, alpha, ptr(A), ptr(B), ptr(C));
    }
}
