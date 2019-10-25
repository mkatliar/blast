#pragma once

#include <blazefeo/SizeT.hpp>
#include <blazefeo/math/typetraits/IsPanelMatrix.hpp>

#include <blaze/math/DenseMatrix.h>

#include <random>


namespace blazefeo
{
    template <typename Derived, size_t P, bool SO = blaze::rowMajor>
    struct PanelMatrix
    :   public blaze::DenseMatrix<Derived, SO>
    {
    };


    template <typename MT, size_t P, bool SO>
    inline typename MT::ElementType * block(PanelMatrix<MT, P, SO>& m, size_t i, size_t j)
    {
        return (~m).block(i, j);
    }


    template <typename MT, size_t P, bool SO>
    inline typename MT::ElementType const * block(PanelMatrix<MT, P, SO> const& m, size_t i, size_t j)
    {
        return (~m).block(i, j);
    }


    template <typename MT, size_t P, bool SO>
    inline size_t spacing(PanelMatrix<MT, P, SO> const& m)
    {
        return (~m).spacing();
    }


    template <typename MT1, typename MT2, typename MT3, size_t P>
    inline auto assign(PanelMatrix<MT1, P, rowMajor>& lhs,
        blaze::DMatDMatMultExpr<MT2, MT3, false, false, false, false> const& rhs)
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && IsPanelMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented");
    }


    template <typename MT1, typename MT2, typename MT3, size_t P>
    inline auto assign(PanelMatrix<MT1, P, rowMajor>& lhs,
        blaze::DMatTDMatMultExpr<MT2, MT3, false, false, false, false> const& rhs)
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && IsRowMajorMatrix_v<MT2> && IsPanelMatrix_v<MT3> && IsRowMajorMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented 2");
    }
}