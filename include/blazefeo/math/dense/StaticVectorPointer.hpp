// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/dense/StorageOrder.hpp>
#include <blazefeo/math/dense/StorageStride.hpp>


namespace blazefeo
{
    template <typename T, size_t S, bool TF, bool AF, bool PF>
    class StaticVectorPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;

        static bool constexpr transposeFlag = TF;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;


        /**
         * @brief Create a pointer pointing to a specified element of a statically-sized vector.
         *
         * @param ptr vector element to be pointed.
         *
         */
        constexpr StaticVectorPointer(T * ptr) noexcept
        :   ptr_ {ptr}
        {
            BLAZE_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        StaticVectorPointer(StaticVectorPointer const&) = default;
        StaticVectorPointer& operator=(StaticVectorPointer const&) = default;


        IntrinsicType load() const noexcept
        {
            return blazefeo::load<AF, SS>(ptr_);
        }


        IntrinsicType maskLoad(MaskType mask) const noexcept
        {
            return blazefeo::maskload(ptr_, mask);
        }


        IntrinsicType broadcast() const noexcept
        {
            return blazefeo::broadcast<SS>(ptr_);
        }


        void store(IntrinsicType val) const noexcept
        {
            blazefeo::store<AF>(ptr_, val);
        }


        void maskStore(MaskType mask, IntrinsicType val) const noexcept
        {
            blazefeo::maskstore(ptr_, mask, val);
        }


        /**
         * @brief Offset pointer by specified number of elements
         *
         * @param i element offset
         *
         * @return offset pointer
         */
        StaticVectorPointer constexpr operator()(ptrdiff_t i) const noexcept
        {
            return {ptrOffset(i)};
        }


        /**
         * @brief Get reference to the pointed value.
         *
         * @return reference to the pointed value
         */
        ElementType& operator*() noexcept
        {
            return *ptr_;
        }


        /**
         * @brief Get const reference to the pointed value.
         *
         * @return const reference to the pointed value
         */
        ElementType& operator*() const noexcept
        {
            return *ptr_;
        }


        /**
        * @brief Convert aligned vector pointer to unaligned.
        */
        StaticVectorPointer<T, S, TF, false, PF> constexpr operator~() const noexcept
        {
            return {ptr_};
        }


        /**
         * @brief Treat row vector as column vector and vise versa.
         *
         * @return transposed vector pointer
         */
        StaticVectorPointer<T, S, !TF, AF, PF> constexpr trans() const noexcept
        {
            return {ptr_};
        }


        /**
         * @brief Distance in memory between adjacent vector elements.
         *
         * @return Distance in memory between adjacent vector elements
         */
        size_t constexpr spacing() const noexcept
        {
            return S;
        }


        /**
         * @brief Get raw pointer
         *
         * @return raw pointer to the vector element
         */
        T * get() const noexcept
        {
            return ptr_;
        }


    private:
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;


        T * ptrOffset(ptrdiff_t i) const noexcept
        {
            return ptr_ + i * spacing();
        }


        T * ptr_;
    };


    template <typename T, size_t S, bool TF, bool AF, bool PF>
    BLAZE_ALWAYS_INLINE auto trans(StaticVectorPointer<T, S, TF, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    template <bool AF, typename VT, bool TF>
    requires (IsStatic_v<VT>)
    BLAZE_ALWAYS_INLINE auto ptr(DenseVector<VT, TF>& v, size_t i)
    {
        if constexpr (IsMajorOriented_v<TF, StorageOrder_v<VT>>)
            return StaticVectorPointer<ElementType_t<VT>, 1, TF, AF, IsPadded_v<VT>> {&(*v)[i]};
        else
            return StaticVectorPointer<ElementType_t<VT>, storageStride_v<VT>, TF, AF, IsPadded_v<VT>> {&(*v)[i]};
    }


    template <bool AF, typename VT, bool TF>
    requires (IsStatic_v<VT>)
    BLAZE_ALWAYS_INLINE auto ptr(DenseVector<VT, TF> const& v, size_t i)
    {
        if constexpr (IsMajorOriented_v<TF, StorageOrder_v<VT>>)
            return StaticVectorPointer<ElementType_t<VT> const, 1, TF, AF, IsPadded_v<VT>> {&(*v)[i]};
        else
            return StaticVectorPointer<ElementType_t<VT> const, storageStride_v<VT>, TF, AF, IsPadded_v<VT>> {&(*v)[i]};
    }


    template <bool AF, typename VT, bool TF>
    requires (IsStatic_v<VT>)
    BLAZE_ALWAYS_INLINE auto ptr(DVecTransExpr<VT, TF> const& v, size_t i)
    {
        if constexpr (IsMajorOriented_v<TF, StorageOrder_v<VT>>)
            return StaticVectorPointer<ElementType_t<VT>, 1, TF, AF, IsPadded_v<VT>> {&(*v)[i]};
        else
            return StaticVectorPointer<ElementType_t<VT>, storageStride_v<VT>, TF, AF, IsPadded_v<VT>> {&(*v)[i]};
    }
}