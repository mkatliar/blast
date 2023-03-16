// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/math/DenseMatrix.h>
#include <blaze/math/DenseVector.h>


namespace blast
{
    /// @brief Solve A*x = b, A lower triangular.
    ///
    template <
        typename MT, bool SO,
        typename VT1, bool TF1,
        typename VT2, bool TF2
    >
    inline void trsvLeftLower(blaze::DenseMatrix<MT, SO> const& A,
        blaze::DenseVector<VT1, TF1> const& b, blaze::DenseVector<VT2, TF2>& x)
    {
        size_t const N = size(b);

        if (rows(A) != N || columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid argument size");

        resize(x, N);

        for (size_t i = 0; i < N; ++i)
        {
           auto acc = (*b)[i];

           for (size_t j = 0; j < i; ++j)
                acc -= (*A)(i, j) * (*x)[j];

           (*x)[i] = acc / (*A)(i, i);
        }
    }


    /// @brief Solve A*x = b, A upper triangular.
    ///
    template <
        typename MT, bool SO,
        typename VT1, bool TF1,
        typename VT2, bool TF2
    >
    inline void trsvLeftUpper(blaze::DenseMatrix<MT, SO> const& A,
        blaze::DenseVector<VT1, TF1> const& b, blaze::DenseVector<VT2, TF2>& x)
    {
        size_t const N = size(b);

        if (rows(A) != N || columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid argument size");

        resize(x, N);

        for (size_t i = N; i-- > 0; )
        {
           auto acc = (*b)[i];

           for (size_t j = i + 1; j < N; ++j)
                acc -= (*A)(i, j) * (*x)[j];

           (*x)[i] = acc / (*A)(i, i);
        }
    }
}