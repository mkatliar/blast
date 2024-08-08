// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#pragma once

#include <blast/math/typetraits/MatrixPointer.hpp>
#include <blast/math/typetraits/VectorPointer.hpp>
#include <blast/util/Types.hpp>


namespace blast :: reference
{
    template <typename Real, typename VPX, typename VPY, typename MPA>
    requires VectorPointer<VPX, Real> && VectorPointer<VPY, Real> && MatrixPointer<MPA, Real>
    inline void ger(size_t m, size_t n, Real alpha, VPX x, VPY y, MPA a)
    {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                *a(i, j) += alpha * *x(i) * *y(j);
    }
}
