// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <boost/preprocessor/iteration/local.hpp>

#include <benchmark/benchmark.h>

#define BENCHMARK_MAX_POTRF 50


namespace blast :: benchmark
{
    using namespace ::benchmark;

    void trmmBenchArguments(internal::Benchmark * b);
}
