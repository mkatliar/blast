// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/ElementType.hpp>

#include <cstddef>


namespace blast
{
    template <typename P, typename T = ElementType_t<P>>
    concept MatrixPointer = requires(P p, std::ptrdiff_t i, std::ptrdiff_t j)
    {
        p(i, j);
        p[i, j];
        p.load();
        p.vmove(i);
        p.hmove(j);
        p.spacing();
        p.trans();
        trans(p);
        ~p;
        *p;
        p.cachePreferredTraversal;

        // {p.get()} -> std::same_as<T *>;
    };
}
