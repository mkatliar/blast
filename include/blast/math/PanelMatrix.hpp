// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/expressions/PanelMatrix.hpp>
#include <blast/math/expressions/PMatTransExpr.hpp>
#include <blast/math/views/Submatrix.hpp>
#include <blast/math/views/Row.hpp>
#include <blast/math/TypeTraits.hpp>


namespace blast
{
    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    inline decltype(auto) submatrix(MT const& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT const, StorageOrder_v<MT>>(matrix, row, column, m, n);
    }


    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    inline decltype(auto) submatrix(MT& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT, StorageOrder_v<MT>>(matrix, row, column, m, n);
    }


    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    inline decltype(auto) row(MT const& matrix, size_t row)
    {
        return PanelMatrixRow<MT const, StorageOrder_v<MT>>(matrix, row);
    }


    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    inline decltype(auto) row(MT& matrix, size_t row)
    {
        return PanelMatrixRow<MT, StorageOrder_v<MT>>(matrix, row);
    }
}
