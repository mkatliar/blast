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

#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/PaddingFlag.h>
#include <blaze/math/TransposeFlag.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsStatic.h>
#include <blaze/math/typetraits/IsSubvector.h>
#include <blaze/math/typetraits/IsTransExpr.h>
#include <blaze/math/typetraits/TransposeFlag.h>
#include <blaze/system/Inline.h>
#include <blazefeo/math/dense/DynamicVectorPointer.hpp>
#include <blazefeo/math/dense/StaticVectorPointer.hpp>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/math/simd/VectorPointer.hpp>
#include <blazefeo/math/simd/MatrixPointer.hpp>
#include <cstddef>


namespace blazefeo
{
    /*
    =======================================================================================

    Pointer to static dense vector.

    =======================================================================================
    */

    template <bool AF, typename ET, size_t N, bool TF, AlignmentFlag AF1, PaddingFlag PF, typename Tag>
    BLAZE_ALWAYS_INLINE auto ptr(StaticVector<ET, N, TF, AF1, PF, Tag>& v, size_t i)
    {
        return StaticVectorPointer<ET, 1, TF, AF, PF> {v.data() + i};
    }


    template <bool AF, typename ET, size_t N, bool TF, AlignmentFlag AF1, PaddingFlag PF, typename Tag>
    BLAZE_ALWAYS_INLINE auto ptr(StaticVector<ET, N, TF, AF1, PF, Tag> const& v, size_t i)
    {
        return StaticVectorPointer<ET const, 1, TF, AF, PF> {v.data() + i};
    }


    /*
    =======================================================================================

    Pointer to dynamic dense vector.

    =======================================================================================
    */

    template <bool AF, typename ET, bool TF, typename Alloc, typename Tag>
    BLAZE_ALWAYS_INLINE auto ptr(DynamicVector<ET, TF, Alloc, Tag>& v, size_t i)
    {
        using VT = DynamicVector<ET, TF, Alloc, Tag>;
        return StaticVectorPointer<ET, 1, TF, AF, IsPadded_v<VT>> {v.data() + i};
    }


    template <bool AF, typename ET, bool TF, typename Alloc, typename Tag>
    BLAZE_ALWAYS_INLINE auto ptr(DynamicVector<ET, TF, Alloc, Tag> const& v, size_t i)
    {
        using VT = DynamicVector<ET, TF, Alloc, Tag>;
        return StaticVectorPointer<ET const, 1, TF, AF, IsPadded_v<VT>> {v.data() + i};
    }


    /*
    =======================================================================================

    Pointer to vector transpose expression.

    =======================================================================================
    */

    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsTransExpr_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT& v, size_t i) noexcept
    {
        return trans(ptr<AF>(v.operand(), i));
    }


    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsTransExpr_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT const& v, size_t i) noexcept
    {
        return trans(ptr<AF>(v.operand(), i));
    }


    /*
    =======================================================================================

    Pointer to row of a matrix.

    =======================================================================================
    */

    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsRow_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT& v, size_t i) noexcept
    {
        using MT = ViewedType_t<VT>;
        using ET = ElementType_t<VT>;

        if constexpr (IsRowMajorMatrix_v<MT>)
        {
            return StaticVectorPointer<ET, 1, rowVector, AF, IsPadded_v<VT>> {&(*v)[i]};
        }
        else
        {
            if constexpr (IsStatic_v<MT>)
                return StaticVectorPointer<ET, MT::spacing(), rowVector, AF, IsPadded_v<VT>> {&(*v)[i]};
            else
                return DynamicVectorPointer<ET, rowVector, AF, IsPadded_v<VT>> {&(*v)[i], v.operand().spacing()};
        }
    }


    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsRow_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT const& v, size_t i) noexcept
    {
        using MT = ViewedType_t<VT>;
        using ET = ElementType_t<VT>;

        if constexpr (IsRowMajorMatrix_v<MT>)
        {
            return StaticVectorPointer<ET const, 1, rowVector, AF, IsPadded_v<VT>> {&(*v)[i]};
        }
        else
        {
            if constexpr (IsStatic_v<MT>)
                return StaticVectorPointer<ET const, MT::spacing(), rowVector, AF, IsPadded_v<VT>> {&(*v)[i]};
            else
                return DynamicVectorPointer<ET const, rowVector, AF, IsPadded_v<VT>> {&(*v)[i], v.operand().spacing()};
        }
    }


    /*
    =======================================================================================

    Pointer to column of a matrix.

    =======================================================================================
    */

    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsColumn_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT& v, size_t i) noexcept
    {
        using MT = ViewedType_t<VT>;
        using ET = ElementType_t<VT>;

        if constexpr (IsColumnMajorMatrix_v<MT>)
        {
            return StaticVectorPointer<ET, 1, columnVector, AF, IsPadded_v<VT>> {&(*v)[i]};
        }
        else
        {
            if constexpr (IsStatic_v<MT>)
                return StaticVectorPointer<ET, MT::spacing(), columnVector, AF, IsPadded_v<VT>> {&(*v)[i]};
            else
                return DynamicVectorPointer<ET, columnVector, AF, IsPadded_v<VT>> {&(*v)[i], v.operand().spacing()};
        }
    }


    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsColumn_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT const& v, size_t i) noexcept
    {
        using MT = ViewedType_t<VT>;
        using ET = ElementType_t<VT>;

        if constexpr (IsColumnMajorMatrix_v<MT>)
        {
            return StaticVectorPointer<ET const, 1, columnVector, AF, IsPadded_v<VT>> {&(*v)[i]};
        }
        else
        {
            if constexpr (IsStatic_v<MT>)
                return StaticVectorPointer<ET const, MT::spacing(), columnVector, AF, IsPadded_v<VT>> {&(*v)[i]};
            else
                return DynamicVectorPointer<ET const, columnVector, AF, IsPadded_v<VT>> {&(*v)[i], v.operand().spacing()};
        }
    }


    /*
    =======================================================================================

    Pointer to a subvector.

    =======================================================================================
    */

    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsSubvector_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT& v, size_t i) noexcept
    {
        return ptr<AF>(v.operand(), v.offset() + i);
    }


    template <bool AF, typename VT>
    requires IsDenseVector_v<VT> && IsSubvector_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT const& v, size_t i) noexcept
    {
        return ptr<AF>(v.operand(), v.offset() + i);
    }


    /*
    =======================================================================================

    Pointer to general dense vector.

    =======================================================================================
    */
    template <typename VT>
    requires IsDenseVector_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT& v)
    {
        return ptr<IsAligned_v<VT>>(v, 0);
    }


    template <typename VT>
    requires IsDenseVector_v<VT>
    BLAZE_ALWAYS_INLINE auto ptr(VT const& v)
    {
        return ptr<IsAligned_v<VT>>(v, 0);
    }


    /**
     * @brief Convert matrix pointer to a column vector pointer.
     *
     * @tparam MP matrix pointer type
     * @param p matrix pointer
     *
     * @return pointer to the matrix column vector whose first element is the one that is pointed by @a p
     */
    template <typename MP>
    requires MatrixPointer<MP>
    BLAZE_ALWAYS_INLINE auto column(MP p) noexcept
    {
        using ET = typename MP::ElementType;

        if constexpr (MP::storageOrder == columnMajor)
        {
            return StaticVectorPointer<ET, 1, columnVector, MP::aligned, MP::padded > {p.get()};
        }
        else
        {
            if constexpr (IsStatic_v<MP>)
            {
                return StaticVectorPointer<ET, MP::spacing(), columnVector, MP::aligned, MP::padded> {p.get()};
            }
            else
            {
                return DynamicVectorPointer<ET, columnVector, MP::aligned, MP::padded> {p.get(), p.spacing()};
            }
        }
    }


    /**
     * @brief Convert matrix pointer to a row vector pointer.
     *
     * @tparam MP matrix pointer type
     * @param p matrix pointer
     *
     * @return pointer to the matrix row vector whose first element is the one that is pointed by @a p
     */
    template <typename MP>
    requires MatrixPointer<MP>
    BLAZE_ALWAYS_INLINE auto row(MP p) noexcept
    {
        using ET = typename MP::ElementType;

        if constexpr (MP::storageOrder == rowMajor)
        {
            return StaticVectorPointer<ET, 1, rowVector, MP::aligned, MP::padded> {p.get()};
        }
        else
        {
            if constexpr (IsStatic_v<MP>)
            {
                return StaticVectorPointer<ET, MP::spacing(), rowVector, MP::aligned, MP::padded> {p.get()};
            }
            else
            {
                return DynamicVectorPointer<ET, rowVector, MP::aligned, MP::padded> {p.get(), p.spacing()};
            }
        }
    }
}