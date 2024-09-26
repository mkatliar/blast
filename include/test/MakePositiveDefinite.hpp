// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//*************************************************************************************************
#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/math/dense/DynamicMatrix.hpp>

#include <stdexcept>


namespace blast :: testing
{
    /**
     * @brief Setup of a random positive definite @a Matrix.
     *
     *
     * @param matrix The matrix to be randomized.
     *
     * @throw @a std::invalid_argument if non-square matrix is provided
     */
    template <Matrix MT>
    inline void makePositiveDefinite(MT& matrix)
    {
        using ET = ElementType_t<MT>;

        size_t const N = columns(matrix);
        if (rows(matrix) != N)
            throw std::invalid_argument {"Non-square matrix provided"};

        DynamicMatrix<ET> tmp(N, N);
        randomize(tmp);

        // TODO: this can be replaced by reference::gemm() when it is there.
        reset(matrix);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < N; ++k)
                    matrix(i, j) += tmp(i, k) * tmp(k, j);

        for (size_t i = 0; i < N; ++i)
            matrix(i, i) += ET(N);
    }
}
