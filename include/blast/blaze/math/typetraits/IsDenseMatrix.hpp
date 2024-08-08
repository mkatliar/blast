// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/IsDenseMatrix.hpp>

#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsMatrix.h>


namespace blast
{
    /**
     * @brief Specialization for Blaze matrices
     *
     * @tparam T matrix type
     */
    template <typename T>
    requires blaze::IsMatrix_v<T>
    struct IsDenseMatrix<T> : blaze::IsDenseMatrix<T> {};
}
