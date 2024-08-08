// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#pragma once

#include <type_traits>


namespace blast
{
    template <typename T>
    struct IsDenseMatrix : std::false_type {};

    template <typename T>
    bool constexpr IsDenseMatrix_v = IsDenseMatrix<T>::value;
}
