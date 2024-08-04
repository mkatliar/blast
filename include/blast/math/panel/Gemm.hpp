// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/PanelMatrix.hpp>
#include <blast/math/panel/MatrixPointer.hpp>
#include <blast/math/algorithm/Gemm.hpp>
#include <blast/util/Exception.hpp>


namespace blast
{
    /**
     * @brief Matrix-matrix multiplication for @a PanelMatrix arguments
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
    template <
        typename ST1, typename MT1, typename MT2, bool SO2,
        typename ST2, typename MT3, typename MT4
    >
    inline void gemm(
        ST1 alpha,
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, SO2> const& B,
        ST2 beta, PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
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
     * @brief Matrix-matrix multiplication for @a PanelMatrix arguments
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
    template <
        typename MT1, typename MT2, bool SO2,
        typename MT3, typename MT4
    >
    inline void gemm(
        PanelMatrix<MT1, columnMajor> const& A, PanelMatrix<MT2, SO2> const& B,
        PanelMatrix<MT3, columnMajor> const& C, PanelMatrix<MT4, columnMajor>& D)
    {
        using ET = ElementType_t<MT4>;
        gemm(ET(1.), A, B, ET(1.), C, D);
    }
}
