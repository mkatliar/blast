// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/ElementType.hpp>

#include <cstddef>


namespace blast
{
    template <typename P, typename T = ElementType_t<P>>
    concept VectorPointer = requires(P p, std::ptrdiff_t i)
    {
        p(i);
        p[i];
        p.load();
        p.broadcast();
        p.spacing();
        p.trans();
        ~p;
        *p;

        // {p.get()} -> std::same_as<T *>;

        P::transposeFlag;
    };
}
