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


namespace blast
{
    /// @brief General matrix-matrix multiplication
    ///
    /// R += alpha * A * B
    ///
    template <typename T, size_t M, size_t N, bool SO, typename PA, typename PB>
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
    inline void gemm(RegisterMatrix<T, M, N, SO>& r, size_t K, T alpha, PA a, PB b) noexcept
    {
        for (size_t k = 0; k < K; ++k)
            r.ger(alpha, column(a(0, k)), row((~b)(k, 0)));
    }


    /// @brief General matrix-matrix multiplication with limited size
    ///
    /// R(0:md-1, 0:nd-1) += alpha * A * B
    ///
    template <typename T, size_t M, size_t N, bool SO, typename PA, typename PB>
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
    inline void gemm(RegisterMatrix<T, M, N, SO>& r, size_t K,
        T alpha, PA a, PB b, size_t md, size_t nd) noexcept
    {
        for (size_t k = 0; k < K; ++k)
            r.ger(alpha, column(a(0, k)), row((~b)(k, 0)), md, nd);
    }


    template <
        typename T, size_t M, size_t N, bool SO,
        typename PA, typename PB, typename PC, typename PD
    >
    requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
        && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    inline void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, T alpha, PA a, PB b, T beta, PC c, PD d) noexcept
    {
        ker.reset();

        for (size_t k = 0; k < K; ++k)
            ker.ger(column(a(0, k)), row((~b)(k, 0)));

        ker *= alpha;
        ker.axpy(beta, c);
        ker.store(d);
    }


    template <
        typename T, size_t M, size_t N, bool SO,
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


    template <
        typename T, size_t M, size_t N, bool SO,
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


    template <
        typename T, size_t M, size_t N, bool SO,
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