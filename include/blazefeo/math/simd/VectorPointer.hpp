// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename P, typename T = ElementType_t<P>>
    concept VectorPointer = requires(P p, ptrdiff_t i)
    {
        p(i);
        p.load();
        p.broadcast();
        p.spacing();
        p.trans();
        trans(p);
        ~p;
        *p;

        // {p.get()} -> std::same_as<T *>;
    };
}