// Copyright 2023-2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/util/Exception.hpp>
#include <blast/system/Tile.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/math/Vector.hpp>


namespace blast
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
        using ET = Scalar;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

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

            // Bottom side
            if (i < M)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                ker.load(ET(1.), A(i, j), M - i, ker.columns());
                ker.ger(alpha, x(i), y(j), M - i, ker.columns());
                ker.store(B(i, j), M - i, ker.columns());
            }
        }


        // Right side
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
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Inconsistent argument sizes"});

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
