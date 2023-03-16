// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <bench/Complexity.hpp>


namespace blast :: benchmark
{
    struct IamaxTag {};
    static IamaxTag constexpr iamaxTag;

    ///
    /// @brief Algorithmic complexity of iamax
    ///
    inline Complexity complexity(IamaxTag, std::size_t n)
    {
        return {
            {"cmp", n - 1},
            {"abs", n}
        };
    }
}


#ifndef BENCHMARK_MAX_IAMAX_STATIC
    #define BENCHMARK_MAX_IAMAX_STATIC 50
#endif

#ifndef BENCHMARK_MAX_IAMAX_DYNAMIC
    #define BENCHMARK_MAX_IAMAX_DYNAMIC 50
#endif