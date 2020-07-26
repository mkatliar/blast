#pragma once

#include <boost/preprocessor/iteration/local.hpp>

#include <benchmark/benchmark.h>

#define BENCHMARK_MAX_SYRK 50
#define BENCHMARK_MAX_GEMM 50
#define BENCHMARK_MAX_POTRF 50


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;
}