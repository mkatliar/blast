// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/util/Types.hpp>


namespace blast :: reference
{
    /**
     * @brief Reference implementation of rank-1 update with multiplier
     *
     * b(i, j) = a(i, j) + alpha * x(i) * y(j)
     * for i=0...m-1, j=n-1
     *
     * @tparam Real real number type
     * @tparam VPX vector pointer type for the column vector @a x
     * @tparam VPY vector pointer type for the row vector @a y
     * @tparam MPA matrix pointer type for the matrix @a a
     * @tparam MPB matrix pointer type for the matrix @a b
     *
     * @param m number of rows in the matrix
     * @param n number of columns in the matrix
     * @param alpha scalar multiplier
     * @param x column vector
     * @param y row vector
     * @param a matrix to perform update on
     *
     */
    template <typename Real, typename VPX, typename VPY, typename MPA, typename MPB>
    requires VectorPointer<VPX, Real> && VectorPointer<VPY, Real> && MatrixPointer<MPA, Real> && MatrixPointer<MPB, Real>
        // && VPX::transposeFlag == columnVector && VPY::transposeFlag == rowVector
    inline void ger(size_t m, size_t n, Real alpha, VPX x, VPY y, MPA a, MPB b)
    {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                *(~b)(i, j) = *(~a)(i, j) + alpha * *(~x)(i) * *(~y)(j);
    }
}
