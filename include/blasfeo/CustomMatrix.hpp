// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo/BlasfeoMatrix.hpp>
#include <blasfeo/Matrix.hpp>


namespace blasfeo
{
    /// @brief BLASFEO matrix using preallocated memory array
    ///
    template <typename Real>
    class CustomMatrix
    :   public BlasfeoMatrix_t<Real>
    ,   public Matrix<CustomMatrix<Real>>
    {
    public:
        using ElementType = Real;


        /// \brief Create a 0-by-0 matrix.
        CustomMatrix()
        {
            create_mat(0, 0, this, nullptr);
        }

        
        CustomMatrix(Real * data, size_t m, size_t n)
        {
            create_mat(m, n, *this, data);
        }


        /// \brief Set new pointer and dimensions.
        /// Use with care!
        void reset(Real * data, size_t m, size_t n)
        {
            create_mat(m, n, *this, data);
        }
    };
}