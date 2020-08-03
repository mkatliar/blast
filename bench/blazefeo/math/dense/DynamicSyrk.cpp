#include <blazefeo/math/dense/Syrk.hpp>

#include <blaze/math/DynamicMatrix.h>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_syrk_dynamic_plain(State& state)
    {
        size_t const M = state.range(0);
        size_t const K = state.range(1);

        DynamicMatrix<Real, columnMajor> A(M, K);
        DynamicMatrix<Real, columnMajor> C(M, M);
        DynamicMatrix<Real, columnMajor> D(M, M);

        randomize(A);
        makeSymmetric(C);

        for (auto _ : state)
        {
            syrkLower(1., A, 1., C, D);
            DoNotOptimize(A);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        setCounters(state.counters, complexitySyrk(M, K));
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_syrk_dynamic_plain, double)->Apply(syrkBenchArguments);
}