// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/StorageOrder.hpp>
#include <blast/math/TransposeFlag.hpp>
#include <blast/math/simd/SimdVec.hpp>
#include <blast/math/simd/SimdMask.hpp>
#include <blast/math/simd/IsSimdAligned.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/util/Assert.hpp>

#include <type_traits>


namespace blast
{
    template <typename T, size_t S, bool SO, bool AF, bool PF>
    class StaticMatrixPointer
    {
    public:
        using ElementType = T;
        using SimdVecType = SimdVec<std::remove_cv_t<T>>;
        using IntrinsicType = SimdVecType::IntrinsicType;
        using MaskType = SimdMask<std::remove_cv_t<T>>;

        static bool constexpr storageOrder = SO;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;
        static bool constexpr isStatic = true;
        static StorageOrder constexpr cachePreferredTraversal = SO == columnMajor ? columnMajor : rowMajor;


        /**
         * @brief Create a pointer pointing to a specified element of a statically-sized matrix.
         *
         * @param p00 pointer to matrix element (0, 0).
         * @param i row index of the pointed element
         * @param j column index of the pointed element
         *
         */
        constexpr StaticMatrixPointer(T * p00, ptrdiff_t i, ptrdiff_t j) noexcept
        :   ptr_ {p00 + (SO == columnMajor ? i + spacing() * j : spacing() * i + j)}
        {
            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr_), "Pointer is not aligned");
        }


        StaticMatrixPointer(StaticMatrixPointer const&) = default;
        StaticMatrixPointer& operator=(StaticMatrixPointer const&) = default;


        SimdVecType load() const noexcept
        {
            return SimdVecType {ptr_, AF};
        }


        SimdVecType load(MaskType mask) const noexcept
        {
            return SimdVecType {ptr_, mask, AF};
        }


        SimdVecType load(TransposeFlag orientation) const
        {
            if (orientation == majorOrientation)
                return SimdVecType {ptr_, AF};
            else
                BLAZE_THROW_LOGIC_ERROR("Cross-load not implemented");
        }


        SimdVecType load(TransposeFlag orientation, MaskType mask) const
        {
            if (orientation == majorOrientation)
                return SimdVecType {ptr_, mask, AF};
            else
                BLAZE_THROW_LOGIC_ERROR("Cross-load not implemented");
        }


        SimdVecType broadcast() const noexcept
        {
            return SimdVecType {*ptr_};
        }


        void store(SimdVecType const& val) const noexcept
        {
            val.store(ptr_, AF);
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
        {
            val.store(ptr_, mask, AF);
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
            return {ptr_, i, j};
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
         * @brief Get reference to the pointed value.
         *
         * @return reference to the pointed value
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
            return {ptr_, 0, 0};
        }


        StaticMatrixPointer<T, S, !SO, AF, PF> constexpr trans() const noexcept
        {
            return {ptr_, 0, 0};
        }


        static size_t constexpr spacing() noexcept
        {
            return S;
        }


        void hmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == columnMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;

            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr_), "Pointer is not aligned");
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;

            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr_), "Pointer is not aligned");
        }


        T * get() const noexcept
        {
            return ptr_;
        }


    private:
        static size_t constexpr SS = SimdVecType::size();
        static TransposeFlag constexpr majorOrientation = SO == columnMajor ? columnVector : rowVector;


        T * ptr_;
    };


    /**
     * @brief Specialization for StaticMatrixPointer
     */
    template <typename T, size_t S, bool SO, bool AF, bool PF>
    struct StorageOrderHelper<StaticMatrixPointer<T, S, SO, AF, PF>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};


    template <typename T, size_t S, bool SO, bool AF, bool PF>
    BLAZE_ALWAYS_INLINE auto trans(StaticMatrixPointer<T, S, SO, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    template <bool AF, Matrix MT>
    requires IsStatic_v<MT> && IsDenseMatrix_v<MT>
    BLAZE_ALWAYS_INLINE auto ptr(MT& m, size_t i, size_t j)
    {
        return StaticMatrixPointer<ElementType_t<MT>, Spacing_v<MT>, StorageOrder_v<MT>, AF, IsPadded_v<MT>>(data(m), i, j);
    }


    template <bool AF, Matrix MT>
    requires IsStatic_v<MT> && IsDenseMatrix_v<MT>
    BLAZE_ALWAYS_INLINE auto ptr(MT const& m, size_t i, size_t j)
    {
        return StaticMatrixPointer<ElementType_t<MT> const, Spacing_v<MT>, StorageOrder_v<MT>, AF, IsPadded_v<MT>>(data(m), i, j);
    }
}
