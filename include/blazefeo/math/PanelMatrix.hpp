// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/expressions/PanelMatrix.hpp>
#include <blazefeo/math/expressions/PMatTransExpr.hpp>
#include <blazefeo/math/views/Submatrix.hpp>
#include <blazefeo/math/views/Row.hpp>


namespace blazefeo
{
    template <typename MT, bool SO>
    inline decltype(auto) submatrix(PanelMatrix<MT, SO> const& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT const, SO>(*matrix, row, column, m, n);
    }


    template <typename MT, bool SO>
    inline decltype(auto) submatrix(PanelMatrix<MT, SO>& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT, SO>(*matrix, row, column, m, n);
    }


    template <typename MT, bool SO>
    inline decltype(auto) row(PanelMatrix<MT, SO> const& matrix, size_t row)
    {
        return PanelMatrixRow<MT const, SO>(*matrix, row);
    }


    template <typename MT, bool SO>
    inline decltype(auto) row(PanelMatrix<MT, SO>& matrix, size_t row)
    {
        return PanelMatrixRow<MT, SO>(*matrix, row);
    }
}