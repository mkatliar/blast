#include <blazefeo/math/dense/Trmm.hpp>

#include <blaze/math/StaticMatrix.h>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M, size_t N>
    static void BM_trmm_static_plain(State& state)
    {
        StaticMatrix<Real, M, M, columnMajor> A;
        StaticMatrix<Real, N, M, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;

        randomize(A);
        randomize(B);

        for (auto _ : state)
        {
            trmm(1., A, trans(B), C);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(M * (M + 1) * N, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
    }


BENCHMARK_TEMPLATE(BM_trmm_static_plain, double, 20, 40);


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(n) BENCHMARK_TEMPLATE(BM_trmm_static_plain, double, n, n);
#include BOOST_PP_LOCAL_ITERATE()
}
