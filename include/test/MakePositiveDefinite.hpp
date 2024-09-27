// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//*************************************************************************************************
#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/math/dense/DynamicMatrix.hpp>
#include <blast/math/reference/Gemm.hpp>
#include <blast/math/expressions/MatTransExpr.hpp>
#include <blast/util/Exception.hpp>


namespace blast :: testing
{
    /**
     * @brief Setup of a random positive definite @a Matrix.
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
            BLAST_THROW_EXCEPTION(std::invalid_argument {"Non-square matrix provided"});

        DynamicMatrix<ET> tmp(N, N);
        randomize(tmp);

        reset(matrix);
        reference::gemm(ET(1), tmp, trans(tmp), ET(0), matrix, matrix);

        for (size_t i = 0; i < N; ++i)
            matrix(i, i) += ET(N);
    }


     /**
     * @brief Setup of a random positive definite matrix,
     * specialized for rvalue reverence to matrix views.
     *
     * @param matrix The matrix to be randomized.
     *
     * @throw @a std::invalid_argument if non-square matrix is provided
     */
    template <Matrix MT>
    requires IsView_v<MT>
    inline void makePositiveDefinite(MT&& matrix)
    {
        makePositiveDefinite(matrix);
    }
}
