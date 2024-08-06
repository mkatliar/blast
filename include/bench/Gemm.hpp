// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <bench/Complexity.hpp>

#define BENCHMARK_MAX_GEMM 50


namespace blast :: benchmark
{
    /// @brief Algorithmic complexity of gemm
    inline Complexity complexityGemm(std::size_t m, std::size_t n, std::size_t k)
    {
        return {
            {"add", (m * n) * (k + 1)},
            {"mul", (m * n) * (k + 2)},
        };
    }
}
