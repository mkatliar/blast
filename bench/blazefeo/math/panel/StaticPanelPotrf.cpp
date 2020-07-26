#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_StaticPanelPotrf(State& state)
    {
        StaticPanelMatrix<Real, M, M, columnMajor> A, L;
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


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_POTRF)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_StaticPanelPotrf, double, n);\
    BENCHMARK_TEMPLATE(BM_StaticPanelPotrf, float, n);
#include BOOST_PP_LOCAL_ITERATE()
}
