// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo/BlasfeoApi.hpp>

#include <blaze/math/DenseMatrix.h>


namespace blasfeo
{
    /// @brief BLASFEO matrix base class
    ///
    template <typename Derived>
    class Matrix
    {
    public:
        Derived& operator~() noexcept
        {
            return static_cast<Derived&>(*this);
        }


        Derived const& operator~() const noexcept
        {
            return static_cast<Derived const&>(*this);
        }


        decltype(auto) operator()(size_t i, size_t j) noexcept
        {
            return element(~*this, i, j);
        }


        decltype(auto) operator()(size_t i, size_t j) const noexcept
        {
            return element(~*this, i, j);
        }


        /// @brief Unpack BLASFEO matrix to Blaze column-major dense matrix.
        ///
        /// The destination matrix is resized if necessary and possible.
        template <typename MT>
        void unpack(blaze::DenseMatrix<MT, blaze::columnMajor>& dst)
        {
            auto const m = rows(~*this);
            auto const n = columns(~*this);

            resize(dst, m, n);
            unpack_mat(m, n, ~*this, 0, 0, data(dst), spacing(dst));
        }


    protected:
        /// \brief Protected constructor to prevent direct instantiation of Matrix objects.
        Matrix()
        {
        }
    };
}