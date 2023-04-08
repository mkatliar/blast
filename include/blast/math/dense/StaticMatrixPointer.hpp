// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/math/StorageOrder.h>
#include <blaze/math/StaticMatrix.h>

#include <blast/math/simd/Simd.hpp>
#include <blast/math/simd/SimdVec.hpp>
#include <blast/math/simd/IsSimdAligned.hpp>
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
        static bool constexpr isStatic = true;


        /**
         * @brief Create a pointer pointing to a specified element of a statically-sized matrix.
         *
         * @param p00 pointer to matrix element (0, 0).
         * @param i row index of the pointed element
         * @param j column index of the pointed element
         *
         */
        constexpr StaticMatrixPointer(T * p00, ptrdiff_t i, ptrdiff_t j) noexcept
        :   ptr_ {ptrOffset(p00, i, j)}
        {
            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr_), "Pointer is not aligned");
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


        SimdVecType broadcast() const noexcept
        {
            return SimdVecType {*ptr_};
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
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;


        static T * ptrOffset(T * ptr, ptrdiff_t i, ptrdiff_t j) noexcept
        {
            if constexpr (SO == columnMajor)
                return ptr + i + spacing() * j;
            else
                return ptr + spacing() * i + j;
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
    BLAZE_ALWAYS_INLINE auto ptr(blaze::DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return StaticMatrixPointer<ElementType_t<MT>, MT::spacing(), SO, AF, IsPadded_v<MT>>((*m).data(), i, j);
    }


    template <bool AF, typename MT, bool SO>
    requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE auto ptr(blaze::DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), SO, AF, IsPadded_v<MT>>((*m).data(), i, j);
    }


    template <bool AF, typename MT, bool SO>
    requires IsStatic_v<MT>
    BLAZE_ALWAYS_INLINE auto ptr(blaze::DMatTransExpr<MT, SO> const& m, size_t i, size_t j)
    {
        return trans(ptr<AF>(m.operand(), j, i));
    }
}