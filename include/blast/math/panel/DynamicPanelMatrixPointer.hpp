// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TransposeFlag.hpp>
#include <blast/math/simd/SimdVec.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/math/simd/Simd.hpp>
#include <blast/math/expressions/PanelMatrix.hpp>
#include <blast/math/expressions/PMatTransExpr.hpp>
#include <blast/util/Assert.hpp>

#include <type_traits>


namespace blast
{
    template <typename T, bool SO, bool AF, bool PF>
    class DynamicPanelMatrixPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;
        using SimdVecType = SimdVec<std::remove_cv_t<T>>;

        static bool constexpr storageOrder = SO;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;
        static bool constexpr isStatic = false;
        static StorageOrder constexpr cachePreferredTraversal = SO == columnMajor ? rowMajor : columnMajor;


        /**
         * @brief Create a pointer pointing to a specified element of a dynamically-sized matrix.
         *
         * @param ptr pointer to matrix element with indices (0, 0).
         * @param spacing stride of the matrix.
         * @param i row index of the pointed element
         * @param j column index of the pointed element
         *
         */
        constexpr DynamicPanelMatrixPointer(T * ptr, size_t spacing, ptrdiff_t i, ptrdiff_t j) noexcept
        :   ptr_ {ptrOffset(ptr, spacing, i, j)}
        ,   spacing_ {spacing}
        {
            BLAST_USER_ASSERT(!AF || isAligned(ptr_), "Pointer is not aligned");
        }


        DynamicPanelMatrixPointer(DynamicPanelMatrixPointer const&) = default;
        DynamicPanelMatrixPointer& operator=(DynamicPanelMatrixPointer const&) = default;


        SimdVecType load() const noexcept
        {
            if constexpr (AF)
                return SimdVecType {ptr_, AF};
            else
            {
                // NOTE: non-optimized!
                std::remove_cv_t<ElementType> tmp[SS];
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
                if constexpr (AF)
                    return SimdVecType {ptr_, AF};
                else
                    static_assert(AF, "load() crossing panel boundary not implemented");
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
            static_assert(AF, "store() implemented only for aligned pointers");
            val.store(ptr_, AF);
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
        {
            static_assert(AF, "store() implemented only for aligned pointers");
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
        DynamicPanelMatrixPointer operator()(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return {ptr_, spacing_, i, j};
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
        DynamicPanelMatrixPointer<T, SO, false, PF> constexpr operator~() const noexcept
        {
            return {ptr_, spacing_, 0, 0};
        }


        DynamicPanelMatrixPointer<T, !SO, AF, PF> constexpr trans() const noexcept
        {
            return {ptr_, spacing_, 0, 0};
        }


        void hmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == columnMajor)
                ptr_ += SimdVecType::size() * inc;
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
                ptr_ += SimdVecType::size() * inc;
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
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;
        static TransposeFlag constexpr majorOrientation = SO == columnMajor ? columnVector : rowVector;


        static T * ptrOffset(T * ptr, size_t spacing, ptrdiff_t i, ptrdiff_t j) noexcept
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
                return ptr + (i / SS) * spacing + i % SS + j * SS;
            else
                return ptr + i * SS + (j / SS) * spacing + j % SS;
        }


        static bool isAligned(T * ptr) noexcept
        {
            return reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0;
        }


        T * ptr_;
        size_t spacing_;
    };


    template <bool SO, typename T, bool AF, bool PF>
    BLAZE_ALWAYS_INLINE auto trans(DynamicPanelMatrixPointer<T, SO, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    template <bool AF, typename MT, bool SO>
    requires (!IsStatic_v<MT>)
    BLAZE_ALWAYS_INLINE auto ptr(PanelMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return DynamicPanelMatrixPointer<ElementType_t<MT>, SO, AF, IsPadded_v<MT>>((*m).data(), spacing(m), i, j);
    }


    template <bool AF, typename MT, bool SO>
    requires (!IsStatic_v<MT>)
    BLAZE_ALWAYS_INLINE auto ptr(PanelMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return DynamicPanelMatrixPointer<ElementType_t<MT> const, SO, AF, IsPadded_v<MT>>((*m).data(), spacing(m), i, j);
    }


    template <bool AF, typename MT, bool SO>
    requires (!IsStatic_v<MT>) && IsPanelMatrix_v<MT>
    BLAZE_ALWAYS_INLINE auto ptr(PMatTransExpr<MT, SO> const& m, size_t i, size_t j)
    {
        return trans(ptr(m.operand(), j, i));
    }
}