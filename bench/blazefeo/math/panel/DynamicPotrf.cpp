#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_potrf_dynamic_panel(State& state)
    {
        size_t const M = state.range(0);

        DynamicPanelMatrix<Real, columnMajor> A(M, M), L(M, M);
        makePositiveDefinite(A);

        for (auto _ : state)
        {
            potrf(A, L);
            DoNotOptimize(A);
            DoNotOptimize(L);
        }

        setCounters(state.counters, complexityPotrf(M, M));
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_potrf_dynamic_panel, double)->DenseRange(1, BENCHMARK_MAX_POTRF);
    BENCHMARK_TEMPLATE(BM_potrf_dynamic_panel, float)->DenseRange(1, BENCHMARK_MAX_POTRF);
}
