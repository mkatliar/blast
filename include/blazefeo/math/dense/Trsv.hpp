#pragma once

#include <blazefeo/Blaze.hpp>


namespace blazefeo
{
    /// @brief Solve A*x = b, A lower triangular.
    ///
    template <
        typename MT, bool SO,
        typename VT1, bool TF1,
        typename VT2, bool TF2
    >
    inline void trsvLeftLower(DenseMatrix<MT, SO> const& A,
        DenseVector<VT1, TF1> const& b, DenseVector<VT2, TF2>& x)
    {
        size_t const N = size(b);

        if (rows(A) != N || columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid argument size");

        resize(x, N);

        for (size_t i = 0; i < N; ++i)
        { 
           auto acc = (~b)[i]; 

           for (size_t j = 0; j < i; ++j)
                acc -= (~A)(i, j) * (~x)[j];

           (~x)[i] = acc / (~A)(i, i);
        } 
    }


    /// @brief Solve A*x = b, A upper triangular.
    ///
    template <
        typename MT, bool SO,
        typename VT1, bool TF1,
        typename VT2, bool TF2
    >
    inline void trsvLeftUpper(DenseMatrix<MT, SO> const& A,
        DenseVector<VT1, TF1> const& b, DenseVector<VT2, TF2>& x)
    {
        size_t const N = size(b);

        if (rows(A) != N || columns(A) != N)
            BLAZE_THROW_INVALID_ARGUMENT("Invalid argument size");

        resize(x, N);

        for (size_t i = N; i-- > 0; )
        { 
           auto acc = (~b)[i]; 

           for (size_t j = i + 1; j < N; ++j)
                acc -= (~A)(i, j) * (~x)[j];

           (~x)[i] = acc / (~A)(i, i);
        } 
    }
}