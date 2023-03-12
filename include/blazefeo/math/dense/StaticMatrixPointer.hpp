// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/dense/StorageOrder.hpp>
#include <blazefeo/math/dense/StorageStride.hpp>


namespace blazefeo
{
    template <typename T, size_t S, bool SO, bool AF, bool PF>
    class StaticMatrixPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;

        static bool constexpr storageOrder = SO;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;


        /**
         * @brief Create a pointer pointing to a specified element of a statically-sized matrix.
         *
         * @param ptr matrix element to be pointed.
         *
         */
        constexpr StaticMatrixPointer(T * ptr) noexcept
        :   ptr_ {ptr}
        {
            BLAZE_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        StaticMatrixPointer(StaticMatrixPointer const&) = default;


        template <typename Other>
        constexpr StaticMatrixPointer(StaticMatrixPointer<Other, S, SO, AF, PF> const& other) noexcept
        :   ptr_ {other.ptr_}
        {
        }


        StaticMatrixPointer& operator=(StaticMatrixPointer const&) = default;


        IntrinsicType load(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return blazefeo::load<AF, SS>(ptrOffset(i, j));
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
            blazefeo::store<AF>(ptrOffset(i, j), val);
        }


        void maskStore(ptrdiff_t i, ptrdiff_t j, MaskType mask, IntrinsicType val) const noexcept
        {
            blazefeo::maskstore(ptrOffset(i, j), mask, val);
        }


        /**
         * @brief Offset pointer by specified number of rows and columns
         *
         * @param i row offset
         * @param j column offset
         *
         * @return offset pointer
         */
        StaticMatrixPointer constexpr operator()(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return {ptrOffset(i, j)};
        }


        StaticMatrixPointer<T, S, !SO, AF, PF> constexpr trans() const noexcept
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


        T * get() const noexcept
        {
            return ptr_;
        }


    private:
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;


        T * ptrOffset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            if constexpr (SO == columnMajor)
                return ptr_ + i + spacing() * j;
            else
                return ptr_ + spacing() * i + j;
        }


        T * ptr_;
    };


    template <typename T, size_t S, bool SO, bool AF, bool PF>
    BLAZE_ALWAYS_INLINE auto trans(StaticMatrixPointer<T, S, SO, AF, PF> const& p) noexcept
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

    template <bool AF, typename MT>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT>, MT::spacing(), columnMajor, AF, IsPadded_v<MT>>
        ptr(DenseMatrix<MT, columnMajor>& m, size_t i, size_t j)
    {
        return {(*m).data() + i + MT::spacing() * j};
    }


    template <bool AF, typename MT>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT>, MT::spacing(), rowMajor, AF, IsPadded_v<MT>>
        ptr(DenseMatrix<MT, rowMajor>& m, size_t i, size_t j)
    {
        return {(*m).data() + MT::spacing() * i + j};
    }


    template <bool AF, typename MT>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), columnMajor, AF, IsPadded_v<MT>>
        ptr(DenseMatrix<MT, columnMajor> const& m, size_t i, size_t j)
    {
        return {(*m).data() + i + MT::spacing() * j};
    }


    template <bool AF, typename MT>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), rowMajor, AF, IsPadded_v<MT>>
        ptr(DenseMatrix<MT, rowMajor> const& m, size_t i, size_t j)
    {
        return {(*m).data() + MT::spacing() * i + j};
    }


    template <bool AF, typename MT>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), columnMajor, AF, IsPadded_v<MT>>
        ptr(DMatTransExpr<MT, columnMajor> const& m, size_t i, size_t j)
    {
        return {(*m).data() + j + MT::spacing() * i};
    }


    template <bool AF, typename MT>
        requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), rowMajor, AF, IsPadded_v<MT>>
        ptr(DMatTransExpr<MT, rowMajor> const& m, size_t i, size_t j)
    {
        return {(*m).data() + MT::spacing() * j + i};
    }


    template <bool AF, typename VT, bool TF>
        requires (IsStatic_v<VT>)
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<VT>, storageStride_v<VT>, StorageOrder_v<VT>, AF, IsPadded_v<VT>>
        ptr(DenseVector<VT, TF>& v, size_t i)
    {
        return {&(*v)[i]};
    }


    template <bool AF, typename VT, bool TF>
        requires (IsStatic_v<VT>)
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<VT> const, storageStride_v<VT>, StorageOrder_v<VT>, AF, IsPadded_v<VT>>
        ptr(DenseVector<VT, TF> const& v, size_t i)
    {
        return {&(*v)[i]};
    }


    template <bool AF, typename VT, bool TF>
        requires (IsStatic_v<VT>)
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<VT> const, storageStride_v<VT>, StorageOrder_v<VT>, AF, IsPadded_v<VT>>
        ptr(DVecTransExpr<VT, TF> const& v, size_t i)
    {
        return {&(*v)[i]};
    }
}