// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/algorithm/Tile.hpp>
#include <blast/math/register_matrix/Gemm.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/util/Exception.hpp>


namespace blast
{
    /**
     * @brief Matrix-matrix multiplication with @a MatrixPointer arguments
     *
     * D := alpha*A*B + beta*C
     *
     * alpha and beta are scalars, and A, B and C are matrices, with A
     * an m by k matrix, B a k by n matrix and C an m by n matrix.
     *
     * @tparam ST1 scalar type for @a alpha
     * @tparam MPA matrix pointer type for @a A
     * @tparam MPB matrix pointer type for @a B
     * @tparam ST2 scalar type for @a beta
     * @tparam MPC matrix pointer type for @a C
     * @tparam MPD matrix pointer type for @a D
     *
     * @param M the number of rows of the matrices A, C, and D.
     * @param N the number of columns of the matrices B and C.
     * @param K the number of columns of the matrix A and the number of rows of the matrix B.
     * @param alpha the scalar alpha
     * @param A the matrix A
     * @param B the matrix B
     * @param beta the scalar beta
     * @param C the matrix C
     * @param D the output matrix D
     */
    template <
        typename ST1, MatrixPointer MPA, MatrixPointer MPB,
        typename ST2, MatrixPointer MPC, MatrixPointer MPD
    >
    inline void gemm(size_t M, size_t N, size_t K, ST1 alpha, MPA A, MPB B, ST2 beta, MPC C, MPD D)
    {
        using ET = std::remove_cv_t<ElementType_t<MPD>>;

        tile<ET, StorageOrder_v<MPD>>(
            xsimd::default_arch {},
            D.cachePreferredTraversal,
            M, N,
            [&] (auto& ker, size_t i, size_t j)
            {
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j));
            },
            [&] (auto& ker, size_t i, size_t j, size_t m, size_t n)
            {
                gemm(ker, K, alpha, A(i, 0), B(0, j), beta, C(i, j), D(i, j), m, n);
            }
        );
    }


    /**
     * @brief Matrix-matrix multiplication for @a DenseMatrix arguments
     *
     * D := alpha*A*B + beta*C
     *
     * alpha and beta are scalars, and A, B and C are matrices, with A
     * an m by k matrix, B a k by n matrix and C an m by n matrix.
     *
     * @param alpha the scalar alpha
     * @param A the matrix A
     * @param B the matrix B
     * @param beta the scalar beta
     * @param C the matrix C
     * @param D the output matrix D
     */
    template <typename ST1, Matrix MT1, Matrix MT2, typename ST2, Matrix MT3, Matrix MT4>
    inline void gemm(ST1 alpha, MT1 const& A, MT2 const& B, ST2 beta, MT3 const& C, MT4& D)
    {
        size_t const M = rows(A);
        size_t const N = columns(B);
        size_t const K = columns(A);

        if (rows(B) != K)
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Matrix sizes do not match"});

        if (rows(C) != M || columns(C) != N)
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Matrix sizes do not match"});

        if (rows(D) != M || columns(D) != N)
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Matrix sizes do not match"});

        gemm(M, N, K, alpha, ptr(*A), ptr(*B), beta, ptr(*C), ptr(*D));
    }


    /**
     * @brief Matrix-matrix multiplication for @a DenseMatrix arguments
     *
     * D := A*B + C
     *
     * A, B and C are matrices, with A
     * an m by k matrix, B a k by n matrix and C an m by n matrix.
     *
     * @param A the matrix A
     * @param B the matrix B
     * @param C the matrix C
     * @param D the output matrix D
     */
    template <Matrix MT1, Matrix MT2, Matrix MT3, Matrix MT4>
    inline void gemm(MT1 const& A, MT2 const& B, MT3 const& C, MT4& D)
    {
        using ET = ElementType_t<MT4>;
        gemm(ET(1.), A, B, ET(1.), C, D);
    }
}
