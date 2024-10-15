// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TransposeFlag.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/math/Simd.hpp>
#include <blast/util/Assert.hpp>
#include <blast/system/Inline.hpp>

#include <type_traits>


namespace blast
{
    template <typename T, bool SO, bool AF, bool PF>
    class DynamicMatrixPointer
    {
    public:
        using ElementType = T;
        using SimdVecType = SimdVec<std::remove_cv_t<T>>;
        using IntrinsicType = SimdVecType::IntrinsicType;
        using MaskType = SimdMask<std::remove_cv_t<T>>;

        static bool constexpr storageOrder = SO;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;
        static bool constexpr isStatic = false;
        static StorageOrder constexpr cachePreferredTraversal = SO == columnMajor ? columnMajor : rowMajor;


        /**
         * @brief Create a pointer pointing to a specified element of a dynamically-sized matrix.
         *
         * @param ptr matrix element to be pointed.
         * @param spacing stride of the matrix.
         *
         */
        constexpr DynamicMatrixPointer(T * ptr, size_t spacing) noexcept
        :   ptr_ {ptr}
        ,   spacing_ {spacing}
        {
            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr), "Pointer is not aligned");
        }


        DynamicMatrixPointer(DynamicMatrixPointer const&) = default;
        DynamicMatrixPointer& operator=(DynamicMatrixPointer const&) = default;


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


        void store(SimdVecType const& val) const noexcept
        {
            val.store(ptr_, AF);
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
        {
            val.store(ptr_, mask, AF);
        }


        size_t spacing() const noexcept
        {
            return spacing_;
        }


        /**
         * @brief Offset pointer by specified number of rows and columns
         *
         * @param i row offset
         * @param j column offset
         *
         * @return offset pointer
         */
        DynamicMatrixPointer operator()(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return {ptrOffset(i, j), spacing_};
        }


        /**
         * @brief Access element at specified offset
         *
         * @param i row offset
         * @param j column offset
         *
         * @return reference to the element at specified offset
         */
        ElementType& operator[](ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return *ptrOffset(i, j);
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
        DynamicMatrixPointer<T, SO, false, PF> constexpr operator~() const noexcept
        {
            return {ptr_, spacing_};
        }


        DynamicMatrixPointer<T, !SO, AF, PF> constexpr trans() const noexcept
        {
            return {ptr_, spacing_};
        }


        void hmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == columnMajor)
                ptr_ += spacing_ * inc;
            else
                ptr_ += inc;

            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr_), "Pointer is not aligned");
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing_ * inc;
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


        T * ptrOffset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            if (SO == columnMajor)
                return ptr_ + i + spacing_ * j;
            else
                return ptr_ + spacing_ * i + j;
        }


        T * ptr_;
        size_t spacing_;
    };


    /**
     * @brief Specialization for @a DynamicMatrixPointer
     */
    template <typename T, bool SO, bool AF, bool PF>
    struct StorageOrderHelper<DynamicMatrixPointer<T, SO, AF, PF>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};


    /**
     * @brief Specialization for @a DynamicMatrixPointer
     */
    template <typename T, bool SO, bool AF, bool PF>
    struct IsAligned<DynamicMatrixPointer<T, SO, AF, PF>> : std::integral_constant<bool, AF> {};


    /**
     * @brief Specialization for @a DynamicMatrixPointer
     */
    template <typename T, bool SO, bool AF, bool PF>
    struct IsPadded<DynamicMatrixPointer<T, SO, AF, PF>> : std::integral_constant<bool, PF> {};


    template <typename T, bool SO, bool AF, bool PF>
    BLAST_ALWAYS_INLINE auto trans(DynamicMatrixPointer<T, SO, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    template <bool AF, Matrix MT>
    requires (!IsStatic_v<MT>) && IsDenseMatrix_v<MT>
    BLAST_ALWAYS_INLINE DynamicMatrixPointer<ElementType_t<MT>, StorageOrder_v<MT>, AF, IsPadded_v<MT>>
        ptr(MT& m, size_t i, size_t j)
    {
        if constexpr (StorageOrder_v<MT> == columnMajor)
            return {data(m) + i + spacing(m) * j, spacing(m)};
        else
            return {data(m) + spacing(m) * i + j, spacing(m)};
    }


    template <bool AF, Matrix MT>
    requires (!IsStatic_v<MT>) && IsDenseMatrix_v<MT>
    BLAST_ALWAYS_INLINE DynamicMatrixPointer<ElementType_t<MT> const, StorageOrder_v<MT>, AF, IsPadded_v<MT>>
        ptr(MT const& m, size_t i, size_t j)
    {
        if constexpr (StorageOrder_v<MT> == columnMajor)
            return {data(m) + i + spacing(m) * j, spacing(m)};
        else
            return {data(m) + spacing(m) * i + j, spacing(m)};
    }
}
