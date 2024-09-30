// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/Matrix.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/algorithm/Gemm.hpp>
#include <blast/math/UpLo.hpp>
#include <blast/system/Tile.hpp>
#include <blast/system/Inline.hpp>
#include <blast/util/Exception.hpp>

#include <stdexcept>


namespace blast
{
    namespace detail
    {
        template <size_t KM, size_t KN, typename T, typename P1, typename P2, typename P3>
        requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T> && MatrixPointer<P3, T>
        BLAST_ALWAYS_INLINE void trmmLeftUpper_backend(size_t M, size_t N, T alpha, P1 a, P2 b, P3 c)
        {
            size_t constexpr TILE_SIZE = TileSize_v<T>;
            static_assert(KM % TILE_SIZE == 0);

            RegisterMatrix<T, KM, KN, columnMajor> ker;

            if (KM <= M)
            {
                size_t j = 0;

                for (; j + KN <= N; j += KN)
                {
                    ker.reset();
                    ker.trmmLeftUpper(alpha, a, b(0, j));
                    gemm(ker, M - KM, alpha, a(0, KM), b(KM, j));
                    ker.store(c(0, j));
                }

                if (j < N)
                {
                    auto const md = KM, nd = N - j;
                    ker.reset();
                    gemm(ker, M, alpha, a, b(0, j), md, nd);
                    ker.store(c(0, j), md, nd);
                }
            }
            else
            {
                // Use partial save to calculate the bottom of the resulting matrix.
                size_t j = 0;

                for (; j + KN <= N; j += KN)
                {
                    auto const md = M, nd = KN;
                    ker.reset();
                    gemm(ker, M, alpha, a, b(0, j), md, nd);
                    ker.store(c(0, j), md, nd);
                }

                if (j < N)
                {
                    auto const md = M, nd = N - j;
                    ker.reset();
                    gemm(ker, M, alpha, a, b(0, j), md, nd);
                    ker.store(c(0, j), md, nd);
                }
            }
        }
    }


    /// @brief C = alpha * A * B; A upper- or lower-triangular. Matrix pointer arguments.
    ///
    /// See https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
    ///
    /// @tparam MPA matrix pointer type for matrix A
    /// @tparam MPB matrix pointer type for matrix B
    /// @tparam MPC matrix pointer type for matrix C
    ///
    /// @param M the number of rows of B
    /// @param N the number of columns of B
    /// @param alpha the scalar alpha
    /// @param A pointer to top left element of matrix A
    /// @param uplo specifies whether the matrix A is an upper or lower triangular
    /// @param diag specifies whether or not A is unit triangular
    /// @param B pointer to top left element of matrix B
    /// @param C pointer to top left element of matrix C
    ///
    template <typename ST, typename MPA, typename MPB, typename MPC>
    requires MatrixPointer<MPA, ST> && MatrixPointer<MPB, ST> && MatrixPointer<MPC, ST>
        && (StorageOrder_v<MPA> == columnMajor) && (StorageOrder_v<MPC> == columnMajor)
    inline void trmm(size_t M, size_t N, ST alpha, MPA A, UpLo uplo, bool diag, MPB B, MPC C)
    {
        using ET = ST;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        if (diag)
            BLAST_THROW_EXCEPTION(std::logic_error {"Unit-triangular matrices support not implemented in trmm()"});

        if (uplo == UpLo::Upper)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 2 * TILE_SIZE < M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
                detail::trmmLeftUpper_backend<3 * TILE_SIZE, TILE_SIZE>(
                    M - i, N, alpha, A(i, i), B(i, 0), C(i, 0));

            for (; i + 1 * TILE_SIZE < M; i += 2 * TILE_SIZE)
                detail::trmmLeftUpper_backend<2 * TILE_SIZE, TILE_SIZE>(
                    M - i, N, alpha, A(i, i), B(i, 0), C(i, 0));

            for (; i + 0 * TILE_SIZE < M; i += 1 * TILE_SIZE)
                detail::trmmLeftUpper_backend<1 * TILE_SIZE, TILE_SIZE>(
                    M - i, N, alpha, A(i, i), B(i, 0), C(i, 0));
        }
        else
        {
            BLAST_THROW_EXCEPTION(std::logic_error {"Left product with lower-triangular matrices not implemented in trmm()"});
        }
    }


    /// @brief C = alpha * B * A; A upper- or lower-triangular. Matrix pointer arguments.
    ///
    /// See https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
    ///
    /// @tparam MPB matrix pointer type for matrix B
    /// @tparam MPA matrix pointer type for matrix A
    /// @tparam MPC matrix pointer type for matrix C
    ///
    /// @param M the number of rows of B
    /// @param N the number of columns of B
    /// @param alpha the scalar alpha
    /// @param B pointer to top left element of matrix B
    /// @param A pointer to top left element of matrix A
    /// @param uplo specifies whether the matrix A is an upper or lower triangular
    /// @param diag specifies whether or not A is unit triangular
    /// @param C pointer to top left element of matrix C
    ///
    template <typename ST, typename MPB, typename MPA, typename MPC>
    requires MatrixPointer<MPB, ST> && MatrixPointer<MPA, ST> && MatrixPointer<MPC, ST>
        && (StorageOrder_v<MPB> == columnMajor) && (StorageOrder_v<MPC> == columnMajor)
    inline void trmm(size_t M, size_t N, ST alpha, MPB B, MPA A, UpLo uplo, bool diag, MPC C)
    {
        using ET = ST;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        if (diag)
            BLAST_THROW_EXCEPTION(std::logic_error {"Unit-triangular matrices support not implemented in trmm()"});

        if (uplo == UpLo::Lower)
        {
            size_t j = 0;

            // Main part
            for (; j + TILE_SIZE <= N; j += TILE_SIZE)
            {
                // size_t const K = N - j - TILE_SIZE;
                size_t i = 0;

                // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
                // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
                for (; i + 3 * TILE_SIZE <= M && i + 4 * TILE_SIZE != M; i += 3 * TILE_SIZE)
                {
                    RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j));
                    /*
                    ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                    ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j));
                    */
                    ker.store(C(i, j));
                }

                for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
                {
                    RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j));
                    /*
                    ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                    ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j));
                    */
                    ker.store(C(i, j));
                }

                for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
                {
                    RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j));
                    /*
                    ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                    ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j));
                    */
                    ker.store(C(i, j));
                }

                // Bottom side
                if (i < M)
                {
                    RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j), M - i, ker.columns());
                    /*
                    ker.trmmRightLower(alpha, ptr<aligned>(B, i, j), ptr<aligned>(A, j, j));
                    ker.gemm(K, alpha, ptr<aligned>(B, i, j + TILE_SIZE), ptr<aligned>(A, j + TILE_SIZE, j), M - i, ker.columns());
                    */
                    ker.store(C(i, j), M - i, ker.columns());
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
                    gemm(ker, N - j, alpha, B(i, j), A(j, j), ker.rows(), N - j);
                    ker.store(C(i, j), ker.rows(), N - j);
                }

                for (; i + 2 * TILE_SIZE <= M; i += 2 * TILE_SIZE)
                {
                    RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j), ker.rows(), N - j);
                    ker.store(C(i, j), ker.rows(), N - j);
                }

                for (; i + 1 * TILE_SIZE <= M; i += 1 * TILE_SIZE)
                {
                    RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j), ker.rows(), N - j);
                    ker.store(C(i, j), ker.rows(), N - j);
                }

                // Bottom-right corner
                if (i < M)
                {
                    RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                    gemm(ker, N - j, alpha, B(i, j), A(j, j), M - i, N - j);
                    ker.store(C(i, j), M - i, N - j);
                }
            }
        }
        else
        {
            BLAST_THROW_EXCEPTION(std::logic_error {"Right product with upper-triangular matrices not implemented in trmm()"});
        }
    }


    /// @brief C = alpha * A * B; A upper- or lower-triangular. Matrix arguments.
    ///
    /// See https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
    ///
    /// @tparam MT1 matrix type for matrix A
    /// @tparam MT2 matrix type for matrix B
    /// @tparam MT3 matrix type for matrix C
    ///
    /// @param alpha the scalar alpha
    /// @param A matrix A
    /// @param uplo specifies whether the matrix A is an upper or lower triangular
    /// @param diag specifies whether or not A is unit triangular
    /// @param B matrix B
    /// @param C matrix C
    ///
    template <typename ST, typename MT1, typename MT2, typename MT3>
    requires Matrix<MT1, ST> && Matrix<MT2, ST> && Matrix<MT3, ST>
    inline void trmm(ST alpha, MT1 const& A, UpLo uplo, bool diag, MT2 const& B, MT3& C)
    {
        using ET = ST;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        size_t const M = rows(B);
        size_t const N = columns(B);

        if (rows(A) != M || columns(A) != M)
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Matrix sizes do not match"});

        if (rows(C) != M || columns(C) != N)
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Matrix sizes do not match"});

        trmm(M, N, alpha, ptr(A), uplo, diag, ptr(B), ptr(C));
    }


    /// @brief C = alpha * B * A + C; A lower-triangular. Matrix arguments.
    ///
    /// See https://netlib.org/lapack/explore-html-3.6.1/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html
    ///
    /// @tparam MTB matrix type for matrix B
    /// @tparam MTA matrix type for matrix A
    /// @tparam MTC matrix type for matrix C
    ///
    /// @param alpha the scalar alpha
    /// @param B matrix B
    /// @param A matrix A
    /// @param uplo specifies whether the matrix A is an upper or lower triangular
    /// @param diag specifies whether or not A is unit triangular
    /// @param C matrix C
    ///
    template <typename ET, typename MTB, typename MTA, typename MTC>
    requires Matrix<MTB, ET> && Matrix<MTA, ET> && Matrix<MTC, ET>
        && (StorageOrder_v<MTB> == columnMajor) && (StorageOrder_v<MTC> == columnMajor)
    inline void trmm(ET alpha, MTB const& B, MTA const& A, UpLo uplo, bool diag, MTC& C)
    {
        size_t const M = rows(B);
        size_t const N = columns(B);

        if (rows(A) != N || columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        if (rows(C) != M || columns(C) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Matrix sizes do not match");

        trmm(M, N, alpha, ptr(B), ptr(A), uplo, diag, ptr(C));
    }
}
