// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/Blaze.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename P, typename T>
    concept MatrixPointer = requires(P p, ptrdiff_t i, ptrdiff_t j)
    {
        p.load(i, j);
        p(i, j);
        p.vmove(i);
        p.hmove(j);
        p.spacing();
        p.trans();
        trans(p);

        // {p.get()} -> std::same_as<T *>;
    };
}