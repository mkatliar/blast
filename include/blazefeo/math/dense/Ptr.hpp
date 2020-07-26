#pragma once

#include <blazefeo/Blaze.hpp>


namespace blazefeo
{   
    template <typename MT, bool SO>
    inline auto * ptr(DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return &(~m)(i, j);
    }


    template <typename MT, bool SO>
    inline auto const * ptr(DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return &(~m)(i, j);
    }
}