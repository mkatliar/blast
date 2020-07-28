#include <bench/Benchmark.hpp>


namespace blazefeo :: benchmark
{
    void syrkBenchArguments(internal::Benchmark * b)
    {
        for (int i = 1; i <= BENCHMARK_MAX_SYRK; ++i)
            b->Args({i, i});
    }


    void trmmBenchArguments(internal::Benchmark * b)
    {
        b->Args({1, 2})->Args({4, 5})->Args({30, 35})->Args({20, 40});
    }
}