#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_syrk(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(M, N);        
        blaze::DynamicMatrix<Real, blaze::columnMajor> B(N, N);

        randomize(A);
        B = 0.;
        
        for (auto _ : state)
            cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, 
                N, M, 1., data(A), spacing(A), 0., data(B), spacing(B));
    }


    static void syrkBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Args({4, 5})->Args({20, 40})->Args({30, 35});
    }


    BENCHMARK_TEMPLATE(BM_syrk, double)->Apply(syrkBenchArguments);
}
