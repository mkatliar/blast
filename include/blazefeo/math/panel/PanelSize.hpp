// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blazefeo/math/simd/Simd.hpp>


namespace blazefeo
{
    template <typename T>
    size_t constexpr PanelSize_v = Simd<T>::size;
}
