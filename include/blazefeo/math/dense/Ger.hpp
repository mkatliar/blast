// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <blazefeo/Exception.hpp>
#include <blazefeo/Blaze.hpp>


namespace blazefeo
{
    /**
     * @brief Performs the rank 1 operation
     *
     *     A := alpha*x*y**T + A,
     *
     * where alpha is a scalar, x is an m element vector, y is an n element
     * vector and A is an m by n column-major matrix.
     *
     * https://netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
     *
     * @tparam Scalar scalar type
     * @tparam VT0 type of first vector
     * @tparam VT1 type of second vector
     * @tparam MT type of matrix
     *
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A matrix
     */
    template <typename Scalar, typename VT0, typename VT1, typename MT>
    inline void ger(Scalar alpha, DenseVector<VT0, columnVector> const& x, DenseVector<VT1, rowVector> const& y, DenseMatrix<MT, columnMajor>&& A)
    {
        size_t const M = size(x);
        size_t const N = size(y);

        if (rows(A) != M || columns(A) != N)
            BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Inconsistent argument sizes"});

        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i < M; ++i)
                (*A)(i, j) += alpha * (*x)[i] * (*y)[j];
    }


    /**
     * @brief Performs the rank 1 operation
     *
     *     A := alpha*x*y**T + A,
     *
     * where alpha is a scalar, x is an m element vector, y is an n element
     * vector and A is an m by n row-major matrix.
     *
     * https://netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
     *
     * @tparam Scalar scalar type
     * @tparam VT0 type of first vector
     * @tparam VT1 type of second vector
     * @tparam MT type of matrix
     *
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A matrix
     */
    template <typename Scalar, typename VT0, typename VT1, typename MT>
    inline void ger(Scalar alpha, DenseVector<VT0, columnVector> const& x, DenseVector<VT1, rowVector> const& y, DenseMatrix<MT, rowMajor>&& A)
    {
        size_t const M = size(x);
        size_t const N = size(y);

        if (rows(A) != M || columns(A) != N)
            BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Inconsistent argument sizes"});

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                (*A)(i, j) += alpha * (*x)[i] * (*y)[j];
    }
}