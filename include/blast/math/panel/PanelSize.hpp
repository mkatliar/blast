// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blast/math/simd/SimdSize.hpp>
#include <blast/math/simd/Simd.hpp>


namespace blast
{
    template <typename T, typename Arch = xsimd::default_arch>
    size_t constexpr PanelSize_v = SimdSize_v<T, Arch>;
}
