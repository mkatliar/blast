// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/IsDenseVector.hpp>

#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsVector.h>


namespace blast
{
    /**
     * @brief Specialization for Blaze vectors
     *
     * @tparam T vector type
     */
    template <typename T>
    requires blaze::IsVector_v<T>
    struct IsDenseVector<T> : blaze::IsDenseVector<T> {};
}
