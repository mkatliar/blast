#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_POTRF_STATIC(type, size) \
    BENCHMARK_TEMPLATE(BM_potrf_static, type, size);

#define BENCHMARK_POTRF_STATIC_10(type, tens) \
    BENCHMARK_POTRF_STATIC(type, tens##0); \
    BENCHMARK_POTRF_STATIC(type, tens##1); \
    BENCHMARK_POTRF_STATIC(type, tens##2); \
    BENCHMARK_POTRF_STATIC(type, tens##3); \
    BENCHMARK_POTRF_STATIC(type, tens##4); \
    BENCHMARK_POTRF_STATIC(type, tens##5); \
    BENCHMARK_POTRF_STATIC(type, tens##6); \
    BENCHMARK_POTRF_STATIC(type, tens##7); \
    BENCHMARK_POTRF_STATIC(type, tens##8); \
    BENCHMARK_POTRF_STATIC(type, tens##9);

#define BENCHMARK_POTRF_STATIC_100(type, hundreds) \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##0); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##1); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##2); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##3); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##4); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##5); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##6); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##7); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##8); \
    BENCHMARK_POTRF_STATIC_10(type, hundreds##9);


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


    BENCHMARK_POTRF_STATIC(double, 1);
    BENCHMARK_POTRF_STATIC(double, 2);
    BENCHMARK_POTRF_STATIC(double, 3);
    BENCHMARK_POTRF_STATIC(double, 4);
    BENCHMARK_POTRF_STATIC(double, 5);
    BENCHMARK_POTRF_STATIC(double, 6);
    BENCHMARK_POTRF_STATIC(double, 7);
    BENCHMARK_POTRF_STATIC(double, 8);
    BENCHMARK_POTRF_STATIC(double, 9);
    BENCHMARK_POTRF_STATIC_10(double, 1);
    BENCHMARK_POTRF_STATIC_10(double, 2);
    BENCHMARK_POTRF_STATIC_10(double, 3);
    BENCHMARK_POTRF_STATIC_10(double, 4);
    BENCHMARK_POTRF_STATIC_10(double, 5);
    // BENCHMARK_POTRF_STATIC_10(double, 6);
    // BENCHMARK_POTRF_STATIC_10(double, 7);
    // BENCHMARK_POTRF_STATIC_10(double, 8);
    // BENCHMARK_POTRF_STATIC_10(double, 9);
    // BENCHMARK_POTRF_STATIC_100(double, 1);
    // BENCHMARK_POTRF_STATIC_100(double, 2);
    // BENCHMARK_POTRF_STATIC(double, 300);
}
