// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/math/StorageOrder.h>
#include <blaze/math/StaticMatrix.h>

#include <blast/math/simd/Simd.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/util/Assert.hpp>


namespace blast
{
    template <typename T, size_t S, bool SO, bool AF, bool PF>
    class StaticMatrixPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;
        using SimdVecType = SimdVec<std::remove_cv_t<T>>;

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
            BLAST_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        StaticMatrixPointer(StaticMatrixPointer const&) = default;
        StaticMatrixPointer& operator=(StaticMatrixPointer const&) = default;


        SimdVecType load() const noexcept
        {
            return SimdVecType {ptr_, AF};
        }


        SimdVecType maskLoad(MaskType mask) const noexcept
        {
            return SimdVecType {ptr_, mask, AF};
        }


        IntrinsicType broadcast() const noexcept
        {
            return blast::broadcast<SS>(ptr_);
        }


        void store(IntrinsicType val) const noexcept
        {
            blast::store<AF>(ptr_, val);
        }


        void maskStore(MaskType mask, IntrinsicType val) const noexcept
        {
            blast::maskstore(ptr_, mask, val);
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
        * @brief Convert aligned matrix pointer to unaligned.
        */
        StaticMatrixPointer<T, S, SO, false, PF> constexpr operator~() const noexcept
        {
            return {ptr_};
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

            BLAST_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr_) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;

            BLAST_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr_) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
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


    template <bool AF, typename MT, bool SO>
    requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT>, MT::spacing(), SO, AF, IsPadded_v<MT>>
        ptr(blaze::DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        if constexpr (SO == columnMajor)
            return {(*m).data() + i + MT::spacing() * j};
        else
            return {(*m).data() + MT::spacing() * i + j};
    }


    template <bool AF, typename MT, bool SO>
    requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), SO, AF, IsPadded_v<MT>>
        ptr(blaze::DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        if constexpr (SO == columnMajor)
            return {(*m).data() + i + MT::spacing() * j};
        else
            return {(*m).data() + MT::spacing() * i + j};
    }


    template <bool AF, typename MT, bool SO>
    requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), SO, AF, IsPadded_v<MT>>
        ptr(blaze::DMatTransExpr<MT, SO> const& m, size_t i, size_t j)
    {
        if constexpr (SO == columnMajor)
            return {(*m).data() + j + MT::spacing() * i};
        else
            return {(*m).data() + MT::spacing() * j + i};
    }
}