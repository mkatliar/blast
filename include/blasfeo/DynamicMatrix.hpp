// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo/BlasfeoMatrix.hpp>
#include <blasfeo/BlasfeoApi.hpp>
#include <blasfeo/Matrix.hpp>

#include <blasfeo/Exception.hpp>

#include <blaze/Math.h>


namespace blasfeo
{
    using namespace blaze;


    /// @brief BLASFEO matrix managing its own memory pool
    ///
    template <typename Real>
    class DynamicMatrix
    :   public BlasfeoMatrix_t<Real>
    ,   public Matrix<DynamicMatrix<Real>>
    {
    public:
        using ElementType = Real;


        /// \brief Create a 0-by-0 matrix.
        DynamicMatrix()
        :   DynamicMatrix(0, 0)
        {
        }


        /// \brief Create a matrix of given size.
        DynamicMatrix(size_t m, size_t n)
        {
            // Use the BLASFEO allocate_mat mechanism, otherwise you can run into this issue:
            // https://github.com/giaf/blasfeo/issues/103
            allocate_mat(m, n, *this);
        }


        /// \brief Create a copy of a Blaze dense column-major matrix.
        template <typename MT>
        DynamicMatrix(blaze::DenseMatrix<MT, blaze::columnMajor> const& rhs)
        :   DynamicMatrix(rows(rhs), columns(rhs))
        {
            pack_mat(rows(rhs), columns(rhs), data(rhs), spacing(rhs), *this, 0, 0);
        }


        ~DynamicMatrix()
        {
            free_mat(*this);
        }


        /// @brief Resize the matrix to new size and re-allocate memory if needed.
        ///
        /// Does not preserve matrix elements if reallocation occurs.
        void resize(size_t m, size_t n)
        {
            if (m != rows(*this) || n != columns(*this))
                BLASFEO_THROW_EXCEPTION(std::invalid_argument("BLASFEO matrix cannot be resized"));
        }


        /// @brief Assign Blaze column-major dense matrix to BLASFEO matrix.
        ///
        /// The BLASFEO matrix is resized if needed.
        template <typename MT>
        DynamicMatrix& operator=(blaze::DenseMatrix<MT, blaze::columnMajor> const& rhs)
        {
            auto const m = rows(rhs);
            auto const n = columns(rhs);

            resize(m, n);
            pack_mat(m, n, data(rhs), spacing(rhs), *this, 0, 0);

            return *this;
        }
    };
}