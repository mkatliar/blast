#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_trmm_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        blaze::LowerMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> A(M, M);        
        blaze::DynamicMatrix<Real, blaze::columnMajor> B(M, N);
        blaze::DynamicMatrix<Real, blaze::columnMajor> C(M, N);

        randomize(A);
        randomize(B);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(C = trans(A) * B);
    }


    static void trmmBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Args({1, 2})->Args({4, 5})->Args({30, 35});
    }


    BENCHMARK_TEMPLATE(BM_trmm_dynamic, double)->Apply(trmmBenchArguments);
}
