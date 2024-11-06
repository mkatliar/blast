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


namespace blast
{
    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    class StaticPanelMatrixPointer
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
        static StorageOrder constexpr cachePreferredTraversal = SO == columnMajor ? rowMajor : columnMajor;


        /**
         * @brief Create a pointer pointing to a specified element of a statically-sized panel matrix.
         *
         * @param ptr pointer to the matrix element with indices (0, 0).
         * @param i row index of the pointed element
         * @param j column index of the pointed element
         *
         */
        constexpr StaticPanelMatrixPointer(T * p00, ptrdiff_t i, ptrdiff_t j) noexcept
        :   ptr_ {ptrOffset(p00, i, j)}
        {
            BLAST_USER_ASSERT(!AF || isAligned(ptr_), "Pointer is not aligned");
        }


        StaticPanelMatrixPointer(StaticPanelMatrixPointer const&) = default;
        StaticPanelMatrixPointer& operator=(StaticPanelMatrixPointer const&) = default;


        SimdVecType load() const noexcept
        {
            if constexpr (AF)
                return SimdVecType {ptr_, AF};
            else
            {
                // NOTE: non-optimized!
                ElementType tmp[SS];
                for (size_t i = 0; i < SS; ++i)
                    tmp[i] = storageOrder == columnMajor ? *(~*this)(i, 0) : *(~*this)(0, i);
                return SimdVecType {tmp, false};
            }
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
            static_assert(AF, "store() implemented only for aligned pointers");
            val.store(ptr_, AF);
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
        {
            static_assert(AF, "store() implemented only for aligned pointers");
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
        StaticPanelMatrixPointer constexpr operator()(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return {ptr_, i, j};
        }


        /**
         * @brief Access element at specified offset
         *
         * @param i row offset
         * @param j column offset
         *
         * @return reference to element at specified offset
         */
        ElementType& operator[](ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return *ptrOffset(ptr_, i, j);
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
        StaticPanelMatrixPointer<T, S, SO, false, PF> constexpr operator~() const noexcept
        {
            return {ptr_, 0, 0};
        }


        StaticPanelMatrixPointer<T, S, !SO, AF, PF> constexpr trans() const noexcept
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
                ptr_ += SS * inc;
            else
            {
                // NOTE: this is correct only for panel-aligned pointers!
                BLAST_USER_ASSERT(isAligned(ptr_), "Pointer is not aligned");
                ptr_ += spacing() * (inc / SS) + inc % SS;
            }

            BLAST_USER_ASSERT(!AF || isAligned(ptr_), "Pointer is not aligned");
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += SS * inc;
            else
            {
                // NOTE: this is correct only for panel-aligned pointers!
                BLAST_USER_ASSERT(isAligned(ptr_), "Pointer is not aligned");
                ptr_ += spacing() * (inc / SS) + inc % SS;
            }

            BLAST_USER_ASSERT(!AF || isAligned(ptr_), "Pointer is not aligned");
        }


        T * get() const noexcept
        {
            return ptr_;
        }


    private:
        static size_t constexpr SS = SimdVecType::size();
        static TransposeFlag constexpr majorOrientation = SO == columnMajor ? columnVector : rowVector;


        static T * ptrOffset(T * ptr, ptrdiff_t i, ptrdiff_t j) noexcept
        {
            if constexpr (!AF)
            {
                auto const rem = reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(ElementType)) / sizeof(ElementType);
                ptr -= rem;

                if constexpr (SO == columnMajor)
                    i += rem;
                else
                    j += rem;
            }

            if constexpr (SO == columnMajor)
                return ptr + (i / SS) * spacing() + i % SS + j * SS;
            else
                return ptr + i * SS + (j / SS) * spacing() + j % SS;
        }


        static bool isAligned(T * ptr) noexcept
        {
            return reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0;
        }


        T * ptr_;
    };


    /**
     * @brief Specialization for @a StaticPanelMatrixPointer
     */
    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    struct StorageOrderHelper<StaticPanelMatrixPointer<T, S, SO, AF, PF>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};


    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    struct IsAligned<StaticPanelMatrixPointer<T, S, SO, AF, PF>> : std::integral_constant<bool, AF> {};


    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    struct IsPadded<StaticPanelMatrixPointer<T, S, SO, AF, PF>> : std::integral_constant<bool, PF> {};


    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    BLAST_ALWAYS_INLINE auto trans(StaticPanelMatrixPointer<T, S, SO, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    template <bool AF, Matrix MT>
    requires IsStatic_v<MT> && IsPanelMatrix_v<MT>
    BLAST_ALWAYS_INLINE auto ptr(MT& m, size_t i, size_t j)
    {
        return StaticPanelMatrixPointer<ElementType_t<MT>, Spacing_v<MT>, StorageOrder_v<MT>, AF, IsPadded_v<MT>>(data(m), i, j);
    }


    template <bool AF, Matrix MT>
    requires IsStatic_v<MT> && IsPanelMatrix_v<MT>
    BLAST_ALWAYS_INLINE auto ptr(MT const& m, size_t i, size_t j)
    {
        return StaticPanelMatrixPointer<ElementType_t<MT> const, Spacing_v<MT>, StorageOrder_v<MT>, AF, IsPadded_v<MT>>(data(m), i, j);
    }
}
