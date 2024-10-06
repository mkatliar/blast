// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/Simd.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/util/Assert.hpp>
#include <blast/system/Inline.hpp>

#include <type_traits>


namespace blast
{
    template <typename T, bool TF, bool AF, bool PF>
    class DynamicVectorPointer
    {
    public:
        using ElementType = T;
        using SimdVecType = SimdVec<std::remove_cv_t<T>>;
        using IntrinsicType = SimdVecType::IntrinsicType;
        using MaskType = SimdMask<std::remove_cv_t<T>>;

        static bool constexpr transposeFlag = TF;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;
        static bool constexpr isStatic = false;


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
            BLAST_USER_ASSERT(spacing > 0, "Vector element spacing must be positive.");
            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr), "Pointer is not aligned");
        }


        DynamicVectorPointer(DynamicVectorPointer const&) = default;
        DynamicVectorPointer& operator=(DynamicVectorPointer const&) = default;


        SimdVecType load() const noexcept
        {
            // Non-optimized
            // TODO: use gather()
            IntrinsicType v;
            for (size_t i = 0; i < SS; ++i)
                v[i] = ptr_[spacing_ * i];

            return SimdVecType {v};
        }


        SimdVecType load(MaskType mask) const noexcept
        {
            // Non-optimized
            // TODO: use gather()
            T v[SS];
            for (size_t i = 0; i < SS; ++i)
                v[i] = mask[i] ? ptr_[spacing_ * i] : T {};

            return SimdVecType {v, false};
        }


        void store(IntrinsicType val) const noexcept
        {
            // Non-optimized
            for (size_t i = 0; i < SS; ++i)
                ptr_[spacing_ * i] = val[i];
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
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
         * @brief Access element at specified offset
         *
         * @param i offset
         *
         * @return reference to the element at specified offset
         */
        ElementType& operator[](ptrdiff_t i) const noexcept
        {
            return *ptrOffset(i);
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
        static size_t constexpr SS = SimdVecType::size();


        T * ptrOffset(ptrdiff_t i) const noexcept
        {
            return ptr_ + i * spacing_;
        }


        T * ptr_;
        size_t spacing_;
    };


    template <bool TF, typename T, bool AF, bool PF>
    BLAST_ALWAYS_INLINE auto trans(DynamicVectorPointer<T, TF, AF, PF> const& p) noexcept
    {
        return p.trans();
    }


    /**
     * @brief Pointer to a dynamically spaced vector
     *
     * @tparam AF true if the pointer is SIMD-aligned
     * @tparam VT vector type
     *
     * @param v vector
     * @param i index within @a v
     *
     * @return vector pointer to @a i -th element of @a v
     */
    template <bool AF, Vector VT>
    requires (!IsStaticallySpaced_v<VT>)
    BLAST_ALWAYS_INLINE auto ptr(VT& v, size_t i) noexcept
    {
        // NOTE: we don't use data(v) here because of this bug:
        // https://bitbucket.org/blaze-lib/blaze/issues/457
        return DynamicVectorPointer<ElementType_t<VT>, VT::transposeFlag, AF, IsPadded_v<VT>> {&v[i], spacing(v)};
    }


    /**
     * @brief Pointer to a dynamically spaced const vector
     *
     * @tparam AF true if the pointer is SIMD-aligned
     * @tparam VT vector type
     *
     * @param v vector
     * @param i index within @a v
     *
     * @return vector pointer to @a i -th element of @a v
     */
    template <bool AF, Vector VT>
    requires (!IsStaticallySpaced_v<VT>)
    BLAST_ALWAYS_INLINE auto ptr(VT const& v, size_t i) noexcept
    {
        // NOTE: we don't use data(v) here because of this bug:
        // https://bitbucket.org/blaze-lib/blaze/issues/457
        return DynamicVectorPointer<ElementType_t<VT> const, VT::transposeFlag, AF, IsPadded_v<VT>> {&v[i], spacing(v)};
    }
}
