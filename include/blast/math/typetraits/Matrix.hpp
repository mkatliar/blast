// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/ElementType.hpp>
#include <blast/util/Types.hpp>


namespace blast
{
    template <typename M, typename T = ElementType_t<M>>
    concept Matrix = requires(M m, T v, T * p, size_t i, size_t j)
    {
        m;
        m(i, j) = v;
        v = m(i, j);
        i = rows(m);
        j = columns(m);
        p = data(m);
    };
}
