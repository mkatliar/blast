// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/UpLo.hpp>


namespace blazefeo
{
    /// @brief Solve A*X = B, with A triangular.
    ///
    template <
        UpLo UPLO,
        bool UNIT,
        typename MT0, bool SO0,
        typename MT1, bool SO1,
        typename MT2, bool SO2
    >
    inline void trsm(DenseMatrix<MT0, SO0> const& A,
        DenseMatrix<MT1, SO1> const& B, DenseMatrix<MT2, SO2>& X)
    {
        size_t const M = rows(B);
        size_t const N = columns(B);

        if (rows(A) != M || columns(A) != M)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid argument size");

        // reize() should be used after this issue is fixed:
        // https://bitbucket.org/blaze-lib/blaze/issues/456/resize-of-a-submatrix
        // resize(X, M, N);
        if (rows(X) != M || columns(X) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid argument size");

        for (size_t j = 0; j < N; ++j)
            if constexpr (UPLO == UpLo::Lower)
                for (size_t i = 0; i < M; ++i)
                {
                    auto acc = (*B)(i, j);

                    for (size_t k = 0; k < i; ++k)
                        acc -= (*A)(i, k) * (*X)(k, j);

                    if constexpr (UNIT)
                        (*X)(i, j) = acc;
                    else
                        (*X)(i, j) = acc / (*A)(i, i);
                }
            else
                for (size_t i = M; i-- > 0; )
                {
                    auto acc = (*B)(i, j);

                    for (size_t k = i + 1; k < M; ++k)
                        acc -= (*A)(i, k) * (*X)(k, j);

                    if constexpr (UNIT)
                        (*X)(i, j) = acc;
                    else
                        (*X)(i, j) = acc / (*A)(i, i);
                }
    }
}