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
#include <blazefeo/system/Tile.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/simd/VectorPointer.hpp>
#include <blazefeo/math/dense/VectorPointer.hpp>
#include <blazefeo/math/dense/MatrixPointer.hpp>


namespace blazefeo
{
    /**
     * @brief Performs the rank 1 operation
     *
     *     B := alpha*x*y**T + A,
     *
     * where alpha is a scalar, x is an m element vector, y is an n element
     * vector and A is an m by n column-major matrix.
     *
     * https://netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
     *
     * @tparam Scalar scalar type
     * @tparam VPX type of first vector
     * @tparam VPY type of second vector
     * @tparam MPA type of input matrix
     * @tparam MPB type of output matrix
     *
     * @param M the number of rows of the matrix A
     * @param N the number of columns of the matrix A
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A input matrix
     * @param B output matrix
     */
    template <typename Scalar, typename VPX, typename VPY, typename MPA, typename MPB>
    requires
        VectorPointer<VPX, Scalar> && (TransposeFlag_v<VPX> == columnVector) &&
        VectorPointer<VPY, Scalar> && (TransposeFlag_v<VPY> == rowVector) &&
        MatrixPointer<MPA, Scalar> && (StorageOrder_v<MPA> == columnMajor) &&
        MatrixPointer<MPB, Scalar> && (StorageOrder_v<MPB> == columnMajor)
    inline void ger(size_t M, size_t N, Scalar alpha, VPX x, VPY y, MPA A, MPB B)
    {
        using ET = ElementType_t<MPB>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<VPX>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<VPY>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MPA>, ET);

        size_t j = 0;

        // Main part
        for (; j + TILE_SIZE <= N; j += TILE_SIZE)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j));
                ker.ger(alpha, x(i), y(j));
                ker.store(B(i, j));
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j));
                ker.ger(alpha, x(i), y(j));
                ker.store(B(i, j));
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j));
                ker.ger(alpha, x(i), y(j));
                ker.store(B(i, j));
            }

            // Bottom edge
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j), M - i, ker.columns());
                ker.ger(alpha, x(i), y(j), M - i, ker.columns());
                ker.store(B(i, j), M - i, ker.columns());
            }
        }


        // Right edge
        if (j < N)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j), ker.rows(), N - j);
                ker.ger(alpha, x(i), y(j), ker.rows(), N - j);
                ker.store(B(i, j), ker.rows(), N - j);
            }

            for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j), ker.rows(), N - j);
                ker.ger(alpha, x(i), y(j), ker.rows(), N - j);
                ker.store(B(i, j), ker.rows(), N - j);
            }

            for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j), ker.rows(), N - j);
                ker.ger(alpha, x(i), y(j), ker.rows(), N - j);
                ker.store(B(i, j), ker.rows(), N - j);
            }

            // Bottom-right corner
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j), M - i, N - j);
                ker.ger(alpha, x(i), y(j), M - i, N - j);
                ker.store(B(i, j), M - i, N - j);
            }
        }
    }


    /**
     * @brief Performs the rank 1 operation
     *
     *     B := alpha*x*y**T + A,
     *
     * where alpha is a scalar, x is an m element vector, y is an n element
     * vector and A is an m by n column-major matrix.
     *
     * https://netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
     *
     * @tparam Scalar scalar type
     * @tparam VT0 type of first vector
     * @tparam VT1 type of second vector
     * @tparam MT0 type of input matrix
     * @tparam MT1 type of output matrix
     *
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A input matrix
     * @param B output matrix
     */
    template <typename Scalar, typename VT0, typename VT1, typename MT0, typename MT1>
    inline void ger(
        Scalar alpha,
        DenseVector<VT0, columnVector> const& x,
        DenseVector<VT1, rowVector> const& y,
        DenseMatrix<MT0, columnMajor> const& A,
        DenseMatrix<MT1, columnMajor>& B
    )
    {
        using ET = ElementType_t<MT1>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<VT0>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<VT1>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<MT0>, ET);

        size_t const M = size(x);
        size_t const N = size(y);

        if (rows(A) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Inconsistent argument sizes");

        if (columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Inconsistent argument sizes");

        if (rows(B) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Inconsistent argument sizes");

        if (columns(B) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Inconsistent argument sizes");

        ger(M, N, alpha, ptr(*x), ptr(*y), ptr(*A), ptr(*B));
    }


    /**
     * @brief Performs the rank 1 operation
     *
     *     B := alpha*x*y**T + A,
     *
     * where alpha is a scalar, x is an m element vector, y is an n element
     * vector and A is an m by n column-major matrix.
     *
     * https://netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
     *
     * @tparam Scalar scalar type
     * @tparam VT0 type of first vector
     * @tparam VT1 type of second vector
     * @tparam MT0 type of input matrix
     * @tparam MT1 type of output matrix
     *
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A input matrix
     * @param B output matrix
     */
    template <typename Scalar, typename VT0, typename VT1, typename MT0, typename MT1>
    inline void ger(
        Scalar alpha,
        DenseVector<VT0, columnVector> const& x,
        DenseVector<VT1, rowVector> const& y,
        DenseMatrix<MT0, columnMajor> const& A,
        DenseMatrix<MT1, columnMajor>&& B
    )
    {
        ger(alpha, x, y, A, B);
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
     * @tparam MT0 type of input matrix
     * @tparam MT1 type of output matrix
     *
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A input matrix
     * @param B output matrix
     */
    template <typename Scalar, typename VT0, typename VT1, typename MT0, typename MT1>
    inline void ger(
        Scalar alpha,
        DenseVector<VT0, columnVector> const& x,
        DenseVector<VT1, rowVector> const& y,
        DenseMatrix<MT0, rowMajor> const& A,
        DenseMatrix<MT1, rowMajor>& B
    )
    {
        size_t const M = size(x);
        size_t const N = size(y);

        if (rows(A) != M || columns(A) != N)
            BLAZEFEO_THROW_EXCEPTION(std::invalid_argument {"Inconsistent argument sizes"});

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                (*B)(i, j) = (*A)(i, j) + alpha * (*x)[i] * (*y)[j];
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
     * @tparam MT0 type of input matrix
     * @tparam MT1 type of output matrix
     *
     * @param alpha scalar alpha
     * @param x first vector
     * @param y second vector
     * @param A input matrix
     * @param B output matrix
     */
    template <typename Scalar, typename VT0, typename VT1, typename MT0, typename MT1>
    inline void ger(
        Scalar alpha,
        DenseVector<VT0, columnVector> const& x,
        DenseVector<VT1, rowVector> const& y,
        DenseMatrix<MT0, rowMajor> const& A,
        DenseMatrix<MT1, rowMajor>&& B
    )
    {
        ger(alpha, x, y, A, B);
    }
}