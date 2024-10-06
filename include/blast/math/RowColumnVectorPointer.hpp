// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/MatrixPointer.hpp>
#include <blast/math/Simd.hpp>
#include <blast/math/TransposeFlag.hpp>
#include <blast/util/Assert.hpp>


namespace blast
{
    template <typename MP, TransposeFlag TF>
    requires MatrixPointer<MP>
    class RowColumnVectorPointer
    {
    public:
        using ElementType = typename MP::ElementType;
        using SimdVecType = SimdVec<std::remove_cv_t<ElementType>>;
        using IntrinsicType = SimdVecType::IntrinsicType;
        using MaskType = SimdMask<std::remove_cv_t<ElementType>>;

        static TransposeFlag constexpr transposeFlag = TF;
        static bool constexpr aligned = MP::aligned;
        static bool constexpr padded = MP::padded;
        static bool constexpr isStatic = MP::isStatic;


        /**
         * @brief Create a row/column pointer from a matrix pointer.
         *
         * @param ptr matrix pointer to vector element 0.
         *
         */
        constexpr RowColumnVectorPointer(MP const& ptr) noexcept
        :   ptr_ {ptr}
        {
        }


        /**
         * @brief Create a row/column pointer from a matrix pointer.
         *
         * @param ptr matrix pointer to vector element 0.
         *
         */
        constexpr RowColumnVectorPointer(MP&& ptr) noexcept
        :   ptr_ {std::move(ptr)}
        {
        }


        RowColumnVectorPointer(RowColumnVectorPointer const&) = default;
        RowColumnVectorPointer& operator=(RowColumnVectorPointer const&) = default;


        SimdVecType load() const noexcept
        {
            return ptr_.load(transposeFlag);
        }


        SimdVecType load(MaskType mask) const noexcept
        {
            return ptr_.load(transposeFlag, mask);
        }


        SimdVecType broadcast() const noexcept
        {
            return ptr_.broadcast();
        }


        void store(SimdVecType const& val) const noexcept
        {
            ptr_.store(transposeFlag, val);
        }


        void store(SimdVecType const& val, MaskType mask) const noexcept
        {
            ptr_.store(transposeFlag, val, mask);
        }


        /**
         * @brief Spacing of the underlying storage.
         *
         * TODO: Do we need it?
         *
         * @return Spacing of the underlying storage
         */
        size_t spacing() const noexcept
        {
            return ptr_.spacing();
        }


        /**
         * @brief Offset pointer by specified number of elements
         *
         * @param i offset
         *
         * @return offset pointer
         */
        RowColumnVectorPointer operator()(ptrdiff_t i) const noexcept
        {
            return transposeFlag == columnVector ? ptr_(i, 0) : ptr_(0, i);
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
            return transposeFlag == columnVector ? ptr_[i, 0] : ptr_[0, i];
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
        * @brief Convert aligned vector pointer to unaligned.
        */
        auto constexpr operator~() const noexcept
        {
            return RowColumnVectorPointer<decltype(~ptr_), transposeFlag> {~ptr_};
        }


        /**
         * @brief Treat row vector as column vector and vise versa.
         *
         * @return transposed vector pointer
         */
        auto constexpr trans() const noexcept
        {
            return RowColumnVectorPointer<decltype(ptr_.trans()), !transposeFlag> {ptr_.trans()};
        }


        /**
         * @brief Get raw pointer
         *
         * @return raw pointer to the vector element
         */
        ElementType * get() const noexcept
        {
            return ptr_.get();
        }


    private:
        static size_t constexpr SS = SimdVecType::size();

        MP ptr_;
    };


    /**
     * @brief Convert matrix pointer to a column vector pointer.
     *
     * @tparam MP matrix pointer type
     * @param p matrix pointer
     *
     * @return pointer to the matrix column vector whose first element is the one that is pointed by @a p
     */
    template <typename MP>
    requires MatrixPointer<MP>
    BLAZE_ALWAYS_INLINE auto column(MP const& p) noexcept
    {
        return RowColumnVectorPointer<MP, columnVector> {p};
    }


    /**
     * @brief Convert matrix pointer to a column vector pointer.
     *
     * @tparam MP matrix pointer type
     * @param p matrix pointer
     *
     * @return pointer to the matrix column vector whose first element is the one that is pointed by @a p
     */
    template <typename MP>
    requires MatrixPointer<MP>
    BLAZE_ALWAYS_INLINE auto column(MP&& p) noexcept
    {
        return RowColumnVectorPointer<MP, columnVector> {std::move(p)};
    }


    /**
     * @brief Convert matrix pointer to a row vector pointer.
     *
     * @tparam MP matrix pointer type
     * @param p matrix pointer
     *
     * @return pointer to the matrix row vector whose first element is the one that is pointed by @a p
     */
    template <typename MP>
    requires MatrixPointer<MP>
    BLAZE_ALWAYS_INLINE auto row(MP const& p) noexcept
    {
        return RowColumnVectorPointer<MP, rowVector> {p};
    }


    /**
     * @brief Convert matrix pointer to a row vector pointer.
     *
     * @tparam MP matrix pointer type
     * @param p matrix pointer
     *
     * @return pointer to the matrix row vector whose first element is the one that is pointed by @a p
     */
    template <typename MP>
    requires MatrixPointer<MP>
    BLAZE_ALWAYS_INLINE auto row(MP&& p) noexcept
    {
        return RowColumnVectorPointer<MP, rowVector> {std::move(p)};
    }
}
