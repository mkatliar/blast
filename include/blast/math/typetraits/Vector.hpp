// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/ElementType.hpp>
#include <blast/util/Types.hpp>


namespace blast
{
    /**
     * @brief Vector concept
     *
     * @tparam V vector type
     * @tparam T element type
     */
    template <typename V, typename T = ElementType_t<V>>
    concept Vector = requires(V v, T a, T * p, size_t i)
    {
        v;
        // v[i] = a;
        a = v[i];
        i = size(v);
        // i = spacing(v);
        p = data(v);
    };
}
