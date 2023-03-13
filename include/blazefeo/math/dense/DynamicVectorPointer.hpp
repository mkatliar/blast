// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/simd/Simd.hpp>


namespace blazefeo
{
    template <typename T, bool TF, bool AF, bool PF>
    class DynamicVectorPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;

        static bool constexpr transposeFlag = TF;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;


        /**
         * @brief Create a pointer pointing to a specified element of a vector with dynamic spacing between elements.
         *
         * @param ptr vector element to be pointed.
         * @param spacing vector element spacing.
         *
         */
        constexpr DynamicVectorPointer(T * ptr, size_t spacing) noexcept
        :   ptr_ {ptr}
        ,   spacing_ {spacing}
        {
            BLAZE_USER_ASSERT(spacing > 0, "Vector element spacing must be positive.");
            BLAZE_USER_ASSERT(!AF || reinterpret_cast<ptrdiff_t>(ptr) % (SS * sizeof(T)) == 0, "Pointer is not aligned");
        }


        DynamicVectorPointer(DynamicVectorPointer const&) = default;
        DynamicVectorPointer& operator=(DynamicVectorPointer const&) = default;


        IntrinsicType load() const noexcept
        {
            // Non-optimized
            IntrinsicType v;
            for (size_t i = 0; i < SS; ++i)
                v[i] = ptr_[spacing_ * i];

            return v;
        }


        IntrinsicType maskLoad(MaskType mask) const noexcept
        {
            // Non-optimized
            IntrinsicType v = blazefeo::setzero<ElementType, SS>();
            for (size_t i = 0; i < SS; ++i)
                if (mask[i])
                    v[i] = ptr_[spacing_ * i];

            return v;
        }


        IntrinsicType broadcast() const noexcept
        {
            return blazefeo::broadcast<SS>(ptr_);
        }


        void store(IntrinsicType val) const noexcept
        {
            // Non-optimized
            for (size_t i = 0; i < SS; ++i)
                ptr_[spacing_ * i] = val[i];
        }


        void maskStore(MaskType mask, IntrinsicType val) const noexcept
        {
            // Non-optimized
            for (size_t i = 0; i < SS; ++i)
                if (mask[i])
                    ptr_[spacing_ * i] = val[i];
        }


        /**
         * @brief Distance in memory between adjacent vector elements.
         *
         * @return Distance in memory between adjacent vector elements
         */
        size_t spacing() const noexcept
        {
            return spacing_;
        }


        /**
         * @brief Offset pointer by specified number of elements
         *
         * @param i offset
         *
         * @return offset pointer
         */
        DynamicVectorPointer operator()(ptrdiff_t i) const noexcept
        {
            return {ptrOffset(i), spacing_};
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
        DynamicVectorPointer<T, TF, false, PF> constexpr operator~() const noexcept
        {
            return {ptr_, spacing_};
        }


        /**
         * @brief Treat row vector as column vector and vise versa.
         *
         * @return transposed vector pointer
         */
        DynamicVectorPointer<T, !TF, AF, PF> constexpr trans() const noexcept
        {
            return {ptr_, spacing_};
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
            return ptr_ + i * spacing_;
        }


        T * ptr_;
        size_t spacing_;
    };


    template <bool TF, typename T, bool AF, bool PF>
    BLAZE_ALWAYS_INLINE auto trans(DynamicVectorPointer<T, TF, AF, PF> const& p) noexcept
    {
        return p.trans();
    }
}