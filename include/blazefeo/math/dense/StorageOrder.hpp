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

#include <blaze/math/StorageOrder.h>
#include <blaze/math/TransposeFlag.h>
#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/dense/StaticVector.h>
#include <blaze/math/typetraits/StorageOrder.h>


namespace blaze
{
    /**
     * @brief Define storage order for dynamic vectors.
     */
    template <typename Scalar, bool TF, typename Alloc, typename Tag>
    bool constexpr StorageOrder_v<DynamicVector<Scalar, TF, Alloc, Tag>> = TF == columnVector ? columnMajor : rowMajor;


    /**
     * @brief Define storage order for static vectors.
     */
    template <typename Scalar, size_t N, bool TF, AlignmentFlag AF, PaddingFlag PF, typename Tag>
    bool constexpr StorageOrder_v<StaticVector<Scalar, N, TF, AF, PF, Tag>> = TF == columnVector ? columnMajor : rowMajor;


    /**
     * @brief Define storage order for matrix rows.
     */
    template <typename MT, bool SO, bool DF, bool SF, size_t... CRAs>
    bool constexpr StorageOrder_v<Row<MT, SO, DF, SF, CRAs...>> = StorageOrder_v<MT>;


    /**
     * @brief Define storage order for matrix columns.
     */
    template <typename MT, bool SO, bool DF, bool SF, size_t... CCAs>
    bool constexpr StorageOrder_v<Column<MT, SO, DF, SF, CCAs...>> = StorageOrder_v<MT>;


    /**
     * @brief Define storage order for subvectors.
     */
    template <typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs >
    bool constexpr StorageOrder_v<Subvector<VT, AF, TF, DF, CSAs...>> = StorageOrder_v<VT>;


    /**
     * @brief Check if a vector orientation matches the major direction
     * of the matrix with the specified storage order.
     *
     * @tparam vector orientation
     * @tparam matrix storage order
     *
     * @return true if @a TF == columnVector and @a SO == columnMajor
     * or @a TF == rowVector and @a SO == rowMajor, false otherwise.
     *
     */
    template <bool TF, bool SO>
    bool constexpr IsMajorOriented_v = TF == columnVector && SO == columnMajor || TF == rowVector && SO == rowMajor;
}