#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    template <typename T>
    void BM_potrf(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        size_t const n = m;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<T, blaze::columnMajor> A(m, m), L(m, m);
        makePositiveDefinite(A);
        
        // Do potrf
        for (auto _ : state)
        {
            L = A;

            int info;
            blaze::potrf('L', m, data(L), spacing(L), &info);
        }

        // Calculated as \sum _{k=0}^{n-1} \sum _{j=0}^{k-1} \sum _{i=k}^{m-1} 2
        state.counters["flops"] = Counter((1 + 3 * m - 2 * n) * (n - 1) * n / 3, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    BENCHMARK_TEMPLATE(BM_potrf, double)->DenseRange(1, 300);
    BENCHMARK_TEMPLATE(BM_potrf, float)->DenseRange(1, 300);
}
