// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


#include <blast/math/Simd.hpp>
#include <blast/util/Assert.hpp>


namespace blast
{
    template <typename T, size_t S, bool TF, bool AF, bool PF>
    class StaticVectorPointer
    {
    public:
        using ElementType = T;
        using SimdVecType = SimdVec<std::remove_cv_t<T>>;
        using IntrinsicType = SimdVecType::IntrinsicType;
        using MaskType = SimdMask<std::remove_cv_t<T>>;

        static bool constexpr transposeFlag = TF;
        static bool constexpr aligned = AF;
        static bool constexpr padded = PF;
        static bool constexpr isStatic = true;


        /**
         * @brief Create a pointer pointing to a specified element of a vector with static spacing between elements.
         *
         * @param ptr vector element to be pointed.
         *
         */
        constexpr StaticVectorPointer(T * ptr) noexcept
        :   ptr_ {ptr}
        {
            BLAST_USER_ASSERT(!AF || isSimdAligned(ptr), "Pointer is not aligned");
        }


        StaticVectorPointer(StaticVectorPointer const&) = default;
        StaticVectorPointer& operator=(StaticVectorPointer const&) = default;


        SimdVecType load() const noexcept
        {
            if constexpr (S == 1)
            {
                return SimdVecType {ptr_, AF};
            }
            else
            {
                // Non-optimized
                IntrinsicType v;
                for (size_t i = 0; i < SS; ++i)
                    v[i] = ptr_[S * i];

                return SimdVecType {v};
            }
        }


        SimdVecType load(MaskType mask) const noexcept
        {
            if constexpr (S == 1)
            {
                return SimdVecType {ptr_, mask, AF};
            }
            else
            {
                // Non-optimized
                // TODO: use gather
                T v[SS];
                for (size_t i = 0; i < SS; ++i)
                    v[i] = mask[i] ? ptr_[S * i] : T {};

                return SimdVecType {v, false};
            }
        }


        SimdVecType broadcast() const noexcept
        {
            return *ptr_;
        }


        void store(SimdVecType val) const noexcept
        {
            if constexpr (S == 1)
            {
                val.store(ptr_, AF);
            }
            else
            {
                // Non-optimized
                for (size_t i = 0; i < SS; ++i)
                    ptr_[S * i] = val[i];
            }
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
        {
            if constexpr (S == 1)
            {
                val.store(ptr_, mask);
            }
            else
            {
                // Non-optimized
                for (size_t i = 0; i < SS; ++i)
                    if (mask[i])
                        ptr_[S * i] = val[i];
            }
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
        static size_t constexpr SS = SimdVecType::size();


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
}
