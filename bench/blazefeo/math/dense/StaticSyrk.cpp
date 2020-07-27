#include <blazefeo/math/dense/Syrk.hpp>

#include <blaze/math/StaticMatrix.h>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M, size_t K>
    static void BM_syrk_static_plain(State& state)
    {
        StaticMatrix<Real, M, K, columnMajor> A;
        StaticMatrix<Real, M, M, columnMajor> C;
        StaticMatrix<Real, M, M, columnMajor> D;

        randomize(A);
        makeSymmetric(C);

        for (auto _ : state)
        {
            syrk_ln(1., A, 1., C, D);
            DoNotOptimize(A);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        setCounters(state.counters, complexitySyrk(M, K));
        state.counters["m"] = M;
    }


    // BENCHMARK_TEMPLATE(BM_syrk_static_plain, double, 40, 20);

#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_SYRK)
#define BOOST_PP_LOCAL_MACRO(n) BENCHMARK_TEMPLATE(BM_syrk_static_plain, double, n, n);
#include BOOST_PP_LOCAL_ITERATE()
}
