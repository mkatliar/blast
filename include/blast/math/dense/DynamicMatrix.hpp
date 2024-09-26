// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/math/simd/SimdSize.hpp>
#include <blast/system/CacheLine.hpp>
#include <blast/system/Restrict.hpp>
#include <blast/util/NextMultiple.hpp>
#include <blast/util/Types.hpp>

#include <new>
#include <stdexcept>


namespace blast
{
    /// @brief Row/column major matrix with dynamically defined size.
    ///
    /// @tparam T element type of the matrix
    /// @tparam SO storage order of panel elements
    ///
    template <typename T, bool SO = columnMajor>
    class DynamicMatrix
    {
    public:
        using ElementType = T;


        explicit DynamicMatrix(size_t m, size_t n)
        :   m_ {m}
        ,   n_ {n}
        ,   spacing_ {nextMultiple(SO == columnMajor ? m : n, SimdSize_v<T>)}
        // Initialize padding elements to 0 to prevent denorms in calculations.
        // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
        ,   v_ {new(std::align_val_t {alignment_}) T[spacing_ * (SO == columnMajor ? n : m)] {}}
        {
        }


        DynamicMatrix(DynamicMatrix const& rhs)
        {
            throw std::logic_error {"Not implemented"};
        }


        DynamicMatrix(DynamicMatrix&& rhs) noexcept
        {
            throw std::logic_error {"Not implemented"};
        }


        ~DynamicMatrix()
        {
            delete[] v_;
        }


        DynamicMatrix& operator=(T val) noexcept
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        DynamicMatrix& operator=(DynamicMatrix const& val)
        {
            throw std::logic_error {"Not implemented"};
            return *this;
        }


        DynamicMatrix& operator=(DynamicMatrix&& val) noexcept
        {
            throw std::logic_error {"Not implemented"};
            return *this;
        }


        T const& operator()(size_t i, size_t j) const noexcept
        {
            return v_[elementIndex(i, j)];
        }


        T& operator()(size_t i, size_t j) noexcept
        {
            return v_[elementIndex(i, j)];
        }


        size_t rows() const noexcept
        {
            return m_;
        }


        size_t columns() const noexcept
        {
            return n_;
        }


        size_t spacing() const noexcept
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
            std::fill_n(v_, spacing_ * (SO == columnMajor ? n_ : m_), T {});
        }


    private:
        static size_t constexpr alignment_ = CACHE_LINE_SIZE;
        static size_t constexpr SS = SimdSize_v<T>;

        size_t m_;
        size_t n_;
        size_t spacing_;
        T * BLAST_RESTRICT v_;


        size_t elementIndex(size_t i, size_t j) const noexcept
        {
            return SO == columnMajor ? i + j * spacing_ : i * spacing_ + j;
        }
    };


    template <typename T, bool SO>
    inline size_t constexpr rows(DynamicMatrix<T, SO> const& m) noexcept
    {
        return m.rows();
    }


    template <typename T, bool SO>
    inline size_t constexpr columns(DynamicMatrix<T, SO> const& m) noexcept
    {
        return m.columns();
    }


    template <typename T, bool SO>
    inline size_t constexpr spacing(DynamicMatrix<T, SO> const& m) noexcept
    {
        return m.spacing();
    }


    template <typename T, bool SO>
    inline constexpr T * data(DynamicMatrix<T, SO>& m) noexcept
    {
        return m.data();
    }


    template <typename T, bool SO>
    inline constexpr T const * data(DynamicMatrix<T, SO> const& m) noexcept
    {
        return m.data();
    }


    template <typename T, bool SO>
    inline void reset(DynamicMatrix<T, SO>& m) noexcept
    {
        m.reset();
    }


    template <typename T, bool SO>
    struct IsDenseMatrix<DynamicMatrix<T, SO>> : std::true_type {};


    template <typename T, bool SO>
    struct IsStatic<DynamicMatrix<T, SO>> : std::false_type {};


    template <typename T, bool SO>
    struct StorageOrderHelper<DynamicMatrix<T, SO>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};
}
