// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/IsAligned.hpp>

#include <blaze/math/typetraits/IsVector.h>
#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/IsAligned.h>


namespace blast
{
    /**
     * @brief Specialization for Blaze matrix and vector types
     *
     * @tparam T matrix or vector type
     */
    template <typename T>
    requires blaze::IsVector_v<T> || blaze::IsMatrix_v<T>
    struct IsAligned<T> : blaze::IsAligned<T> {};
}
