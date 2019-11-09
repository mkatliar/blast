#pragma once

#include <blazefeo/math/typetraits/IsPanelMatrix.hpp>

#include <blaze/math/ReductionFlag.h>
#include <blaze/math/Matrix.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Types.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename Derived, bool SO = rowMajor>
    struct PanelMatrix
    :   public Matrix<Derived, SO>
    {
    };


    template <typename MT, bool SO>
    inline decltype(auto) tile(PanelMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return (~m).tile(i, j);
    }


    template <typename MT, bool SO>
    inline decltype(auto) tile(PanelMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return (~m).tile(i, j);
    }


    template <typename MT, bool SO>
    inline auto * ptr(PanelMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return (~m).ptr(i, j);
    }


    template <typename MT, bool SO>
    inline auto const * ptr(PanelMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return (~m).ptr(i, j);
    }


    template <typename MT, bool SO>
    inline size_t spacing(PanelMatrix<MT, SO> const& m)
    {
        return (~m).spacing();
    }


    template <typename MT1, typename MT2, typename MT3, bool SO>
    inline auto assign(PanelMatrix<MT1, SO>& lhs,
        blaze::DMatDMatMultExpr<MT2, MT3, false, false, false, false> const& rhs)
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && IsPanelMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented");
    }


    template <typename MT1, typename MT2, typename MT3, bool SO>
    inline auto assign(PanelMatrix<MT1, SO>& lhs,
        blaze::DMatTDMatMultExpr<MT2, MT3, false, false, false, false> const& rhs)
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && IsRowMajorMatrix_v<MT2> && IsPanelMatrix_v<MT3> && IsRowMajorMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented 2");
    }
}