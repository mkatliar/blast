// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/StorageOrder.hpp>
#include <blast/math/Simd.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/system/CacheLine.hpp>
#include <blast/util/NextMultiple.hpp>
#include <blast/util/Types.hpp>

#include <initializer_list>
#include <type_traits>


namespace blast
{
    /// @brief Matrix with statically defined size.
    ///
    /// @tparam T element type of the matrix
    /// @tparam M number of rows
    /// @tparam N number of columns
    /// @tparam SO storage order
    template <typename T, size_t M, size_t N, StorageOrder SO>
    class StaticMatrix
    {
    public:
        using ElementType = T;
        static bool constexpr storageOrder = SO;


        StaticMatrix() noexcept
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_, capacity_, T {});
        }


        StaticMatrix(T const& v) noexcept
        {
            std::fill_n(v_, capacity_, v);
        }


        /**
         * @brief Construct from an initializer list.
         *
         * The initializer list is row-major regardless of the @a RegsterMatrix storage order,
         * such that the initializer elements are written in a natural way.
         *
         * \code
         * StaticMatrix<double, 2, 3, columnMajor> m {
         *     {1., 2., 3.},
         *     {4., 5., 6.},
         *     {7., 8., 9.}
         * };
         * \endcode
         *
         * @param list list of lists of matrix row elements
         *
         */
        constexpr StaticMatrix(std::initializer_list<std::initializer_list<T>> list)
        {
            std::fill_n(v_, capacity_, T {});

            if (list.size() != M || determineColumns(list) > N)
                throw std::invalid_argument {"Invalid setup of static matrix"};

            size_t i = 0;

            for (auto const& row : list)
            {
                size_t j = 0;

                for (const auto& element : row)
                {
                    v_[elementIndex(i, j)] = element;
                    ++j;
                }

                ++i;
            }
        }


        StaticMatrix& operator=(T val) noexcept
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        constexpr T const& operator()(size_t i, size_t j) const noexcept
        {
            return v_[elementIndex(i, j)];
        }


        constexpr T& operator()(size_t i, size_t j)
        {
            return v_[elementIndex(i, j)];
        }


        static size_t constexpr rows() noexcept
        {
            return M;
        }


        static size_t constexpr columns() noexcept
        {
            return N;
        }


        static size_t constexpr spacing() noexcept
        {
            return spacing_;
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
         * @brief Set all matrix elements to 0
         */
        void reset() noexcept
        {
            std::fill_n(v_, capacity_, T {});
        }


    private:
        static size_t constexpr spacing_ = nextMultiple(SO == columnMajor ? M : N, SimdSize_v<T>);
        static size_t constexpr capacity_ = spacing_ * (SO == columnMajor ? N : M);

        // Alignment of the data elements.
        static size_t constexpr alignment_ = CACHE_LINE_SIZE;

        // Aligned element storage.
        alignas(alignment_) T v_[capacity_];


        size_t elementIndex(size_t i, size_t j) const
        {
            return SO == columnMajor ? i + spacing_ * j : spacing_ * i + j;
        }
    };


    template <typename T, size_t M, size_t N, StorageOrder SO>
    inline size_t constexpr rows(StaticMatrix<T, M, N, SO> const& m) noexcept
    {
        return m.rows();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    inline size_t constexpr columns(StaticMatrix<T, M, N, SO> const& m) noexcept
    {
        return m.columns();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    inline constexpr T * data(StaticMatrix<T, M, N, SO>& m) noexcept
    {
        return m.data();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    inline constexpr T const * data(StaticMatrix<T, M, N, SO> const& m) noexcept
    {
        return m.data();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    inline void reset(StaticMatrix<T, M, N, SO>& m) noexcept
    {
        m.reset();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    struct IsDenseMatrix<StaticMatrix<T, M, N, SO>> : std::true_type {};


    template <typename T, size_t M, size_t N, StorageOrder SO>
    struct IsStatic<StaticMatrix<T, M, N, SO>> : std::true_type {};


    template <typename T, size_t M, size_t N, StorageOrder SO>
    struct Spacing<StaticMatrix<T, M, N, SO>> : std::integral_constant<size_t, StaticMatrix<T, M, N, SO>::spacing()> {};


    template <typename T, size_t M, size_t N, StorageOrder SO>
    struct StorageOrderHelper<StaticMatrix<T, M, N, SO>> : std::integral_constant<StorageOrder, SO> {};


    template <typename T, size_t M, size_t N, StorageOrder SO>
    struct IsAligned<StaticMatrix<T, M, N, SO>> : std::true_type {};


    template <typename T, size_t M, size_t N, StorageOrder SO>
    struct IsPadded<StaticMatrix<T, M, N, SO>> : std::true_type {};
}
