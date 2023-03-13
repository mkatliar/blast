// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/simd/Simd.hpp>


namespace blazefeo
{
    template <typename T, bool SO, bool AF, bool PF>
    class DynamicMatrixPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;

        static bool constexpr storageOrder = SO;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;


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
            BLAZE_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        DynamicMatrixPointer(DynamicMatrixPointer const&) = default;
        DynamicMatrixPointer& operator=(DynamicMatrixPointer const&) = default;


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

            BLAZE_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr_) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing_ * inc;
            else
                ptr_ += inc;

            BLAZE_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr_) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        T * get() const noexcept
        {
            return ptr_;
        }


    private:
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;


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


    template <bool SO, typename T, bool AF, bool PF>
    BLAZE_ALWAYS_INLINE auto trans(DynamicMatrixPointer<T, SO, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    template <bool AF, typename MT, bool SO>
        requires (!IsStatic_v<MT>)
    BLAZE_ALWAYS_INLINE DynamicMatrixPointer<ElementType_t<MT>, SO, AF, IsPadded_v<MT>>
        ptr(DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        if constexpr (SO == columnMajor)
            return {(*m).data() + i + spacing(m) * j, spacing(m)};
        else
            return {(*m).data() + spacing(m) * i + j, spacing(m)};
    }


    template <bool AF, typename MT, bool SO>
        requires (!IsStatic_v<MT>)
    BLAZE_ALWAYS_INLINE DynamicMatrixPointer<ElementType_t<MT> const, SO, AF, IsPadded_v<MT>>
        ptr(DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        if constexpr (SO == columnMajor)
            return {(*m).data() + i + spacing(m) * j, spacing(m)};
        else
            return {(*m).data() + spacing(m) * i + j, spacing(m)};
    }


    template <bool AF, typename MT, bool SO>
        requires (!IsStatic_v<MT>)
    BLAZE_ALWAYS_INLINE DynamicMatrixPointer<ElementType_t<MT> const, SO, AF, IsPadded_v<MT>>
        ptr(DMatTransExpr<MT, SO> const& m, size_t i, size_t j)
    {
        if constexpr (SO == columnMajor)
            return {(*m).data() + i + spacing(m) * j, spacing(m)};
        else
            return {(*m).data() + spacing(m) * i + j, spacing(m)};
    }


    template <bool AF, bool PF, bool SO, typename T>
    BLAZE_ALWAYS_INLINE DynamicMatrixPointer<T, SO, AF, PF> ptr(T * p, size_t spacing)
    {
        return {p, spacing};
    }
}