// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <blaze/math/typetraits/IsAligned.h>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/math/simd/MatrixPointer.hpp>


namespace blazefeo
{
    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE auto ptr(DenseMatrix<MT, SO>& m)
    {
        return ptr<IsAligned_v<MT>>(m, 0, 0);
    }


    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE auto ptr(DenseMatrix<MT, SO> const& m)
    {
        return ptr<IsAligned_v<MT>>(m, 0, 0);
    }


    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE auto ptr(DMatTransExpr<MT, SO> const& m)
    {
        return ptr<IsAligned_v<MT>>(m, 0, 0);
    }
}