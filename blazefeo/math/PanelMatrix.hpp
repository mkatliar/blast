#pragma once

#include <blazefeo/math/expressions/PanelMatrix.hpp>
#include <blazefeo/math/expressions/PMatTransExpr.hpp>
#include <blazefeo/math/views/Submatrix.hpp>


namespace blazefeo
{
    template <typename MT, bool SO>
    inline decltype(auto) submatrix(PanelMatrix<MT, SO> const& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT const, SO>(~matrix, row, column, m, n);
    }


    template <typename MT, bool SO>
    inline decltype(auto) submatrix(PanelMatrix<MT, SO>& matrix, size_t row, size_t column, size_t m, size_t n)
    {
        return PanelSubmatrix<MT, SO>(~matrix, row, column, m, n);
    }
}