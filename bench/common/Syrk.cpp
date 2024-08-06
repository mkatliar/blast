// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Syrk.hpp>


namespace blast :: benchmark
{
    void syrkBenchArguments(internal::Benchmark * b)
    {
        for (int i = 1; i <= BENCHMARK_MAX_SYRK; ++i)
            b->Args({i, i});
    }


    Complexity complexitySyrk(std::size_t m, std::size_t k)
    {
        return {
            {"add", m * (m + 1) * k / 2},
            {"mul", m * (m + 1) * k / 2}
        };
    }
}
