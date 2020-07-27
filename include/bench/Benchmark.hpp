#pragma once

#include <boost/preprocessor/iteration/local.hpp>

#include <benchmark/benchmark.h>

#define BENCHMARK_MAX_SYRK 50
#define BENCHMARK_MAX_GEMM 50
#define BENCHMARK_MAX_POTRF 50


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    inline void syrkBenchArguments(internal::Benchmark * b) 
    {
        for (int i = 1; i <= BENCHMARK_MAX_SYRK; ++i)
            b->Args({i, i});
    }
}