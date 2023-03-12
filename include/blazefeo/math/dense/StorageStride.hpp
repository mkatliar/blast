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

#include <blaze/math/typetraits/IsDenseVector.h>
#include <blazefeo/Blaze.hpp>


namespace blazefeo
{
    /**
     * @brief Storage stride of a dynamically-sized dense vector.
     *
     * @tparam VT vector type
     * @tparam TF transform flag
     *
     * @param v vector
     */
    template <typename VT, bool TF>
    requires (!IsStatic_v<VT>)
    inline size_t storageStride(DenseVector<VT, TF> const& v)
    {
        return (*v).spacing();
    }


    /**
     * @brief Storage stride of a dynamically-sized dense subvector.
     *
     * It equals the spacing of the underlying vector.
     *
     * @tparam VT underlying vector type
     * @tparam AF Alignment flag
     * @tparam TF Transpose flag
     * @tparam DF Density flag
     * @tparam CSAs Compile time subvector arguments
     *
     * @param v vector
     */
    template <typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs>
    requires (!IsStatic_v<VT> && IsDenseVector_v<VT>)
    inline size_t storageStride(Subvector<VT, AF, TF, DF, CSAs...> const& v)
    {
        return v.operand().spacing();
    }


    /**
     * @brief Storage stride of a statically-sized dense vector type.
     *
     * @tparam VT vector type
     */
    template <typename VT>
    requires (IsStatic_v<VT> && IsDenseVector_v<VT>)
    size_t constexpr storageStride_v = VT::spacing();


    /**
     * @brief Storage stride of a statically-sized dense matrix row type.
     *
     * It equals the spacing of the underlying matrix.
     *
     * @tparam MT underlying matrix type
     */
    template <typename MT, bool SO, bool DF, bool SF, size_t... CRAs>
    requires (IsStatic_v<MT>)
    size_t constexpr storageStride_v<Row<MT, SO, DF, SF, CRAs...>> = MT::spacing();


    /**
     * @brief Storage stride of a statically-sized dense matrix column type.
     *
     * It equals the spacing of the underlying matrix.
     *
     * @tparam MT underlying matrix type
     */
    template <typename MT, bool SO, bool DF, bool SF, size_t... CRAs>
    requires (IsStatic_v<MT>)
    size_t constexpr storageStride_v<Column<MT, SO, DF, SF, CRAs...>> = MT::spacing();


    /**
     * @brief Storage stride of a statically-sized subvector type.
     *
     * It equals the spacing of the underlying vector.
     *
     * @tparam VT underlying vector type
     */
    template <typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs>
    requires (IsStatic_v<VT>)
    size_t constexpr storageStride_v<Subvector<VT, AF, TF, DF, CSAs...>> = storageStride_v<VT>;
}