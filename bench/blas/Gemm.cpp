#include <blaze/Math.h>

#include <benchmark/benchmark.h>

#include <vector>


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    template <typename Real>
    static void BM_gemm(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        blaze::DynamicMatrix<Real, blaze::columnMajor> A(m, m);
        randomize(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(m, m);
        randomize(B);

        blaze::DynamicMatrix<Real, blaze::columnMajor> C(m, m);
        randomize(C);
        
        for (auto _ : state)
            gemm(C, trans(A), B, 1.0, 1.0);

        state.counters["flops"] = Counter(2 * m * m * m, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }
    
    BENCHMARK_TEMPLATE(BM_gemm, double)->DenseRange(1, 50);
    BENCHMARK_TEMPLATE(BM_gemm, float)->DenseRange(1, 50);
}
