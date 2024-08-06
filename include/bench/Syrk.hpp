// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <bench/Complexity.hpp>

#define BENCHMARK_MAX_SYRK 20


namespace blast :: benchmark
{
    void syrkBenchArguments(internal::Benchmark * b);

    /// @brief Algorithmic complexity of syrk
    Complexity complexitySyrk(std::size_t m, std::size_t k);
}
