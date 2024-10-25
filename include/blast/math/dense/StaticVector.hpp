// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/Forward.hpp>
#include <blast/math/TransposeFlag.hpp>
#include <blast/math/Simd.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/system/CacheLine.hpp>
#include <blast/util/NextMultiple.hpp>
#include <blast/util/Types.hpp>

#include <initializer_list>
#include <type_traits>


namespace blast
{
    /// @brief Vector with statically defined size.
    ///
    /// @tparam T element type of the vector
    /// @tparam N number of elements
    /// @tparam TF transpose flag
    template <typename T, size_t N, TransposeFlag TF>
    class StaticVector
    {
    public:
        using ElementType = T;
        static TransposeFlag constexpr transposeFlag = TF;


        StaticVector() noexcept
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_, capacity_, T {});
        }


        StaticVector(T const& v) noexcept
        {
            std::fill_n(v_, capacity_, v);
        }


        /**
         * @brief Construct from an initializer list.
         *
         * \code
         * StaticVector<double, 3, columnVector> v {1., 2., 3.};
         * \endcode
         *
         * @param list list of vector elements. If @a list is shorter than @a N,
         * the remaining vector elements will be 0. If @a list is longer than @a N,
         * the extra elements of @a list will be ignored.
         */
        constexpr StaticVector(std::initializer_list<T> list)
        {
            fill(copy_n(list.begin(), std::min(list.size(), N), std::begin(v_)), std::end(v_), T {});
        }


        StaticVector& operator=(T val) noexcept
        {
            std::fill_n(v_, capacity_, val);

            return *this;
        }


        constexpr T const& operator[](size_t i) const noexcept
        {
            assert(i < N);
            return v_[i];
        }


        constexpr T& operator[](size_t i)
        {
            assert(i < N);
            return v_[i];
        }


        static size_t constexpr size() noexcept
        {
            return N;
        }


        T * data() noexcept
        {
            return v_;
        }


        T const * data() const noexcept
        {
            return v_;
        }


        /**
         * @brief Set all vector elements to 0
         */
        void reset() noexcept
        {
            std::fill_n(v_, capacity_, T {});
        }


    private:
        static size_t constexpr capacity_ = nextMultiple(N, SimdSize_v<T>);

        // Alignment of the data elements.
        static size_t constexpr alignment_ = CACHE_LINE_SIZE;

        // Aligned element storage.
        alignas(alignment_) T v_[capacity_];

    };


    template <typename T, size_t N, TransposeFlag TF>
    inline size_t constexpr size(StaticVector<T, N, TF> const& m) noexcept
    {
        return N;
    }


    template <typename T, size_t N, TransposeFlag TF>
    inline constexpr T * data(StaticVector<T, N, TF>& m) noexcept
    {
        return m.data();
    }


    template <typename T, size_t N, TransposeFlag TF>
    inline constexpr T const * data(StaticVector<T, N, TF> const& m) noexcept
    {
        return m.data();
    }


    template <typename T, size_t N, TransposeFlag TF>
    inline void reset(StaticVector<T, N, TF>& m) noexcept
    {
        m.reset();
    }


    template <typename T, size_t N, TransposeFlag TF>
    struct IsDenseVector<StaticVector<T, N, TF>> : std::true_type {};


    template <typename T, size_t N, TransposeFlag TF>
    struct IsStatic<StaticVector<T, N, TF>> : std::true_type {};


    template <typename T, size_t N, TransposeFlag TF>
    struct IsAligned<StaticVector<T, N, TF>> : std::integral_constant<bool, true> {};


    template <typename T, size_t N, TransposeFlag TF>
    struct IsPadded<StaticVector<T, N, TF>> : std::integral_constant<bool, true> {};


    template <typename T, size_t N, TransposeFlag TF>
    struct IsStaticallySpaced<StaticVector<T, N, TF>> : std::integral_constant<bool, true> {};


    template <typename T, size_t N, TransposeFlag TF>
    struct Spacing<StaticVector<T, N, TF>> : std::integral_constant<size_t, 1> {};
}
