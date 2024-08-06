// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Benchmark.hpp>


namespace blast :: benchmark
{
    void trmmBenchArguments(internal::Benchmark * b)
    {
        b->Args({1, 2})->Args({4, 5})->Args({30, 35})->Args({20, 40});
    }
}
