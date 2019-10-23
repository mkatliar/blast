#include <smoke/DynamicPanelMatrix.hpp>
#include <smoke/Gemm.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace smoke :: benchmark
{
    template <typename Real>
    static void BM_gemm_nt_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = M;
        size_t const K = M;

        DynamicPanelMatrix<Real> A(M, K);
        DynamicPanelMatrix<Real> B(N, K);
        DynamicPanelMatrix<Real> C(M, N);
        DynamicPanelMatrix<Real> D(M, N);

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
        {
            gemm_nt(A, B, C, D);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }
    

    BENCHMARK_TEMPLATE(BM_gemm_nt_dynamic, double)->DenseRange(1, 50);
}
