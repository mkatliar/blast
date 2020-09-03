// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/simd/Simd.hpp>


namespace blazefeo
{
    template <typename T, size_t S, bool SO>
    class StaticMatrixPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;

        static bool constexpr storageOrder = SO;

        
        constexpr StaticMatrixPointer(T * ptr) noexcept
        :   ptr_ {ptr}
        {
        }


        StaticMatrixPointer(StaticMatrixPointer const&) = default;


        template <typename Other>
        constexpr StaticMatrixPointer(StaticMatrixPointer<Other, S, SO> const& other) noexcept
        :   ptr_ {other.ptr_}
        {
        }


        StaticMatrixPointer& operator=(StaticMatrixPointer const&) = default;


        IntrinsicType load(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return blazefeo::load<SS>(ptrOffset(i, j));
        }


        IntrinsicType maskLoad(ptrdiff_t i, ptrdiff_t j, MaskType mask) const noexcept
        {
            return blazefeo::maskload(ptrOffset(i, j), mask);
        }


        IntrinsicType broadcast(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return blazefeo::broadcast<SS>(ptrOffset(i, j));
        }


        void store(ptrdiff_t i, ptrdiff_t j, IntrinsicType val) const noexcept
        {
            blazefeo::store(ptrOffset(i, j), val);
        }


        void maskStore(ptrdiff_t i, ptrdiff_t j, MaskType mask, IntrinsicType val) const noexcept
        {
            blazefeo::maskstore(ptrOffset(i, j), mask, val);
        }


        StaticMatrixPointer constexpr offset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return {ptrOffset(i, j)};
        }


        StaticMatrixPointer<T, S, !SO> constexpr trans() const noexcept
        {
            return {ptr_};
        }


        size_t constexpr spacing() const noexcept
        {
            return S;
        }


        void hmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == columnMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;
        }


    private:
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;

        
        T * ptrOffset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            if (SO == columnMajor)
                return ptr_ + i + spacing() * j;
            else
                return ptr_ + spacing() * i + j;
        }


        T * ptr_;
    };


    template <typename T, size_t S, bool SO>
    BLAZE_ALWAYS_INLINE auto trans(StaticMatrixPointer<T, S, SO> const& p) noexcept
    {
        return p.trans();
    }


    // NOTE:
    // IsStatic_v<...> for adapted static matrix types such as 
    // e.g. SymmetricMatrix<StaticMatrix<...>> evaluates to false;
    // therefore ptr() for these types will return a DynamicMatrixPointer,
    // which is not performance-optimal.
    //
    // See this issue: https://bitbucket.org/blaze-lib/blaze/issues/368
    //

    template <typename MT, bool SO>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT>, MT::spacing(), SO>
        ptr(DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return {&(*m)(i, j)};
    }


    template <typename MT, bool SO>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), SO>
        ptr(DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return {&(*m)(i, j)};
    }


    template <typename MT, bool SO>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), SO>
        ptr(DMatTransExpr<MT, SO> const& m, size_t i, size_t j)
    {
        return {&(*m)(i, j)};
    }
}