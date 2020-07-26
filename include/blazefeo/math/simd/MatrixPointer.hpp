#pragma once

#include <blazefeo/Blaze.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename Derived, bool SO>
    class MatrixPointerBase
    {
    protected:
        MatrixPointerBase() = default;
        MatrixPointerBase(MatrixPointerBase const&) = default;
    };


    template <typename P, bool SO>
    concept MatrixPointer = std::is_base_of_v<MatrixPointerBase<P, SO>, P>;
}