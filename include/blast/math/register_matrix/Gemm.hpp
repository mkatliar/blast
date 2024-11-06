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

#include <blast/math/register_matrix/RegisterMatrix.hpp>
#include <blast/math/TypeTraits.hpp>


namespace blast
{
    /// @brief General matrix-matrix multiplication performed in-place
    ///
    /// R += alpha * A * B,
    /// where R is M by N, A is M by K, and B is K by N.
    ///
    template <typename T, size_t M, size_t N, StorageOrder SO, typename PA, typename PB>
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
    inline void gemm(RegisterMatrix<T, M, N, SO>& r, size_t K, T alpha, PA a, PB b) noexcept
    {
        for (size_t k = 0; k < K; ++k)
            r.ger(alpha, column(a(0, k)), row((~b)(k, 0)));
    }


    /// @brief General matrix-matrix multiplication for a sub-matrix performed in-place
    ///
    /// R(0:md-1, 0:nd-1) += alpha * A * B,
    /// where R is M by N, A is md by K, and B is K by nd.
    ///
    template <typename T, size_t M, size_t N, StorageOrder SO, typename PA, typename PB>
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
    inline void gemm(RegisterMatrix<T, M, N, SO>& r, size_t K,
        T alpha, PA a, PB b, size_t md, size_t nd) noexcept
    {
        for (size_t k = 0; k < K; ++k)
            r.ger(alpha, column(a(0, k)), row((~b)(k, 0)), md, nd);
    }


    /// @brief General matrix-matrix multiplication
    ///
    /// D = alpha * A * B + beta * C,
    /// where D and C are M by N, A is M by K, and B is K by N.
    ///
    /// The @a RegisterMatrix @a ker is used for intermediate calculations and has undefined value on return.
    ///
    /// TODO: the @a ker argument could be removed and M, N passed as the function template parameters.
    /// T and SO could be inferred from the argument types.
    ///
    template <
        typename T, size_t M, size_t N, StorageOrder SO,
        typename PA, typename PB, typename PC, typename PD
    >
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
        && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    inline void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, T alpha, PA a, PB b, T beta, PC c, PD d) noexcept
    {
        ker.reset();

        bool constexpr a_columns_aligned = IsAligned_v<PA> && IsPadded_v<PA> && StorageOrder_v<PA> == columnMajor;
        bool constexpr b_rows_aligned = IsAligned_v<PB> && IsPadded_v<PB> && StorageOrder_v<PB> == rowMajor;

        for (size_t k = 0; k < K; ++k)
            if constexpr (a_columns_aligned && b_rows_aligned)
                ker.ger(column(a(0, k)), row(b(k, 0)));
            else if constexpr (a_columns_aligned && !b_rows_aligned)
                ker.ger(column(a(0, k)), row((~b)(k, 0)));
            else if constexpr (!a_columns_aligned && b_rows_aligned)
                ker.ger(column((~a)(0, k)), row(b(k, 0)));
            else
                ker.ger(column((~a)(0, k)), row((~b)(k, 0)));

        ker *= alpha;
        ker.axpy(beta, c);
        ker.store(d);
    }


    /// @brief General matrix-matrix multiplication for a sub-matrix
    ///
    /// D = alpha * A * B + beta * C,
    /// where D and C are M by N, A is M by K, and B is K by N.
    ///
    /// The @a RegisterMatrix @a ker is used for intermediate calculations and has undefined value on return.
    ///
    /// TODO: the @a ker argument could be removed and M, N passed as the function template parameters.
    /// T and SO could be inferred from the argument types.
    ///
    template <
        typename T, size_t M, size_t N, StorageOrder SO,
        typename PA, typename PB, typename PC, typename PD
    >
    requires
        MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) &&
        MatrixPointer<PB, T> &&
        MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    inline void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, T alpha, PA a, PB b, T beta, PC c, PD d, size_t md, size_t nd) noexcept
    {
        ker.reset();

        for (size_t k = 0; k < K; ++k)
            ker.ger(column(a(0, k)), row((~b)(k, 0)), md, nd);

        ker *= alpha;
        ker.axpy(beta, c, md, nd);
        ker.store(d, md, nd);
    }


    /// @brief Matrix-matrix multiplication
    ///
    /// D = alpha * A * B + beta * C,
    /// where D and C are M by N, A is M by K, and B is K by N.
    ///
    /// The @a RegisterMatrix @a ker is used for intermediate calculations and has undefined value on return.
    ///
    /// TODO: the @a ker argument could be removed and M, N passed as the function template parameters.
    /// T and SO could be inferred from the argument types.
    ///
    template <
        typename T, size_t M, size_t N, StorageOrder SO,
        typename PA, typename PB, typename PC, typename PD
    >
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
        && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    inline void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, PA a, PB b, PC c, PD d) noexcept
    {
        ker.load(c);

        for (size_t k = 0; k < K; ++k)
            ker.ger(column(a(0, k)), row((~b)(k, 0)));

        ker.store(d);
    }


    /// @brief Matrix-matrix multiplication for a sub-matrix
    ///
    /// D = alpha * A * B + beta * C,
    /// where D and C are md by nd, A is md by K, and B is K by nd.
    ///
    /// The @a RegisterMatrix @a ker is used for intermediate calculations and has undefined value on return.
    ///
    /// TODO: the @a ker argument could be removed and M, N passed as the function template parameters.
    /// T and SO could be inferred from the argument types.
    ///
    template <
        typename T, size_t M, size_t N, StorageOrder SO,
        typename PA, typename PB, typename PC, typename PD
    >
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
        && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    inline void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, PA a, PB b, PC c, PD d, size_t md, size_t nd) noexcept
    {
        ker.load(c);

        for (size_t k = 0; k < K; ++k)
            ker.ger(column(a(0, k)), row((~b)(k, 0)), md, nd);

        ker.store(d, md, nd);
    }
}
