#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_POTRF_STATIC(size) \
    BENCHMARK_TEMPLATE(BM_potrf_static, double, size);\
    BENCHMARK_TEMPLATE(BM_potrf_static, float, size);

#define BENCHMARK_POTRF_STATIC_10(tens) \
    BENCHMARK_POTRF_STATIC(tens##0); \
    BENCHMARK_POTRF_STATIC(tens##1); \
    BENCHMARK_POTRF_STATIC(tens##2); \
    BENCHMARK_POTRF_STATIC(tens##3); \
    BENCHMARK_POTRF_STATIC(tens##4); \
    BENCHMARK_POTRF_STATIC(tens##5); \
    BENCHMARK_POTRF_STATIC(tens##6); \
    BENCHMARK_POTRF_STATIC(tens##7); \
    BENCHMARK_POTRF_STATIC(tens##8); \
    BENCHMARK_POTRF_STATIC(tens##9);

#define BENCHMARK_POTRF_STATIC_100(hundreds) \
    BENCHMARK_POTRF_STATIC_10(hundreds##0); \
    BENCHMARK_POTRF_STATIC_10(hundreds##1); \
    BENCHMARK_POTRF_STATIC_10(hundreds##2); \
    BENCHMARK_POTRF_STATIC_10(hundreds##3); \
    BENCHMARK_POTRF_STATIC_10(hundreds##4); \
    BENCHMARK_POTRF_STATIC_10(hundreds##5); \
    BENCHMARK_POTRF_STATIC_10(hundreds##6); \
    BENCHMARK_POTRF_STATIC_10(hundreds##7); \
    BENCHMARK_POTRF_STATIC_10(hundreds##8); \
    BENCHMARK_POTRF_STATIC_10(hundreds##9);


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_potrf_static(::benchmark::State& state)
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


    BENCHMARK_POTRF_STATIC(1);
    BENCHMARK_POTRF_STATIC(2);
    BENCHMARK_POTRF_STATIC(3);
    BENCHMARK_POTRF_STATIC(4);
    BENCHMARK_POTRF_STATIC(5);
    BENCHMARK_POTRF_STATIC(6);
    BENCHMARK_POTRF_STATIC(7);
    BENCHMARK_POTRF_STATIC(8);
    BENCHMARK_POTRF_STATIC(9);
    BENCHMARK_POTRF_STATIC_10(1);
    BENCHMARK_POTRF_STATIC_10(2);
    BENCHMARK_POTRF_STATIC_10(3);
    BENCHMARK_POTRF_STATIC_10(4);
    BENCHMARK_POTRF_STATIC_10(5);
    BENCHMARK_POTRF_STATIC_10(6);
    BENCHMARK_POTRF_STATIC_10(7);
    BENCHMARK_POTRF_STATIC_10(8);
    BENCHMARK_POTRF_STATIC_10(9);
    BENCHMARK_POTRF_STATIC_100(1);
    BENCHMARK_POTRF_STATIC_100(2);
    BENCHMARK_POTRF_STATIC(300);
}
