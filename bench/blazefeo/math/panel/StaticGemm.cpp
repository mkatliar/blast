#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <bench/Gemm.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm_static_panel(State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticPanelMatrix<Real, M, K> A;
        StaticPanelMatrix<Real, N, K> B;
        StaticPanelMatrix<Real, M, N> C;
        StaticPanelMatrix<Real, M, N> D;

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


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_gemm_static_panel, double, n); \
    BENCHMARK_TEMPLATE(BM_gemm_static_panel, float, n);
#include BOOST_PP_LOCAL_ITERATE()
}
