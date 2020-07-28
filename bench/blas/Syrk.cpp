#include <blazefeo/Blaze.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_syrk(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const K = state.range(1);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(M, K);        
        blaze::DynamicMatrix<Real, blaze::columnMajor> B(M, M);

        randomize(A);
        B = 0.;
        
        for (auto _ : state)
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, 
                M, K, 1., data(A), spacing(A), 1., data(B), spacing(B));

        setCounters(state.counters, complexitySyrk(M, K));
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_syrk, double)->Apply(syrkBenchArguments);
        // ->Args({4, 5})->Args({20, 40})->Args({30, 35})
}
