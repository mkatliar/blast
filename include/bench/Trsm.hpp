// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#define BENCHMARK_MAX_TRSM 50

#include <bench/Complexity.hpp>


namespace blast :: benchmark
{
    struct TrsmTag {};
    static TrsmTag constexpr trsmTag;


    /**
     * @brief Algorithmic complexity of triangular substitution.
     *
     * See https://algowiki-project.org/en/Backward_substitution
     *
     * @param unit true if the matrix is unit-diagonal
     * @param m number of rows and columns in the triangular matrix
     * @param n number of columns in matrix the second matrix
     *
     * @return Complexity
     */
    inline Complexity complexity(TrsmTag, bool unit, size_t m, size_t n)
    {
        return {
            {"add", ((m * m - m) / 2) * n},
            {"mul", ((m * m - m) / 2) * n},
            {"div", unit ? 0 : m}
        };
    }
}