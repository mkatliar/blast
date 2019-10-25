#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm_nt_static(::benchmark::State& state)
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


    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 1);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 2);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 3);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 4);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 5);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 6);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 7);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 8);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 9);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 10);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 11);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 12);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 13);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 14);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 15);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 16);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 17);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 18);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 19);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 20);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 21);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 22);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 23);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 24);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 25);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 26);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 27);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 28);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 29);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 30);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 31);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 32);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 33);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 34);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 35);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 36);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 37);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 38);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 39);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 40);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 41);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 42);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 43);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 44);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 45);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 46);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 47);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 48);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 49);
    BENCHMARK_TEMPLATE(BM_gemm_nt_static, double, 50);
}
