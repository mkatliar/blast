#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <bench/Gemm.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_gemm_dynamic_panel(State& state)
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
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }

    
    BENCHMARK_TEMPLATE(BM_gemm_dynamic_panel, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
    BENCHMARK_TEMPLATE(BM_gemm_dynamic_panel, float)->DenseRange(1, BENCHMARK_MAX_GEMM);
}
