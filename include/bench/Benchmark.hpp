// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <boost/preprocessor/iteration/local.hpp>

#include <benchmark/benchmark.h>

#define BENCHMARK_MAX_SYRK 50
#define BENCHMARK_MAX_POTRF 50
#define BENCHMARK_MAX_GETRF 50


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    void syrkBenchArguments(internal::Benchmark * b);
    void trmmBenchArguments(internal::Benchmark * b);
}