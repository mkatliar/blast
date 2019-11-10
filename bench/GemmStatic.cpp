#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_GEMM_NT_STATIC(type, size) \
    BENCHMARK_TEMPLATE(BM_gemm_nt_static_inline, type, size); \
    BENCHMARK_TEMPLATE(BM_gemm_nt_static_noinline, type, size);


namespace blazefeo :: benchmark
{
    template <typename MT1, bool SO1, typename MT2, bool SO2, typename MT3, bool SO3, typename MT4, bool SO4>
    __attribute((noinline)) static void gemm_nt_noinline(
        Matrix<MT1, SO1> const& A, Matrix<MT2, SO2> const& B, 
        Matrix<MT3, SO3> const& C, Matrix<MT4, SO4>& D)
    {
        gemm_nt(~A, ~B, ~C, ~D);
    }


    template <typename Real, size_t M>
    static void BM_gemm_nt_static_inline(::benchmark::State& state)
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


    template <typename Real, size_t M>
    static void BM_gemm_nt_static_noinline(::benchmark::State& state)
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
            gemm_nt_noinline(A, B, C, D);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    BENCHMARK_GEMM_NT_STATIC(double, 1);
    BENCHMARK_GEMM_NT_STATIC(double, 2);
    BENCHMARK_GEMM_NT_STATIC(double, 3);
    BENCHMARK_GEMM_NT_STATIC(double, 4);
    BENCHMARK_GEMM_NT_STATIC(double, 5);
    BENCHMARK_GEMM_NT_STATIC(double, 6);
    BENCHMARK_GEMM_NT_STATIC(double, 7);
    BENCHMARK_GEMM_NT_STATIC(double, 8);
    BENCHMARK_GEMM_NT_STATIC(double, 9);
    BENCHMARK_GEMM_NT_STATIC(double, 10);
    BENCHMARK_GEMM_NT_STATIC(double, 11);
    BENCHMARK_GEMM_NT_STATIC(double, 12);
    BENCHMARK_GEMM_NT_STATIC(double, 13);
    BENCHMARK_GEMM_NT_STATIC(double, 14);
    BENCHMARK_GEMM_NT_STATIC(double, 15);
    BENCHMARK_GEMM_NT_STATIC(double, 16);
    BENCHMARK_GEMM_NT_STATIC(double, 17);
    BENCHMARK_GEMM_NT_STATIC(double, 18);
    BENCHMARK_GEMM_NT_STATIC(double, 19);
    BENCHMARK_GEMM_NT_STATIC(double, 20);
    BENCHMARK_GEMM_NT_STATIC(double, 21);
    BENCHMARK_GEMM_NT_STATIC(double, 22);
    BENCHMARK_GEMM_NT_STATIC(double, 23);
    BENCHMARK_GEMM_NT_STATIC(double, 24);
    BENCHMARK_GEMM_NT_STATIC(double, 25);
    BENCHMARK_GEMM_NT_STATIC(double, 26);
    BENCHMARK_GEMM_NT_STATIC(double, 27);
    BENCHMARK_GEMM_NT_STATIC(double, 28);
    BENCHMARK_GEMM_NT_STATIC(double, 29);
    BENCHMARK_GEMM_NT_STATIC(double, 30);
    BENCHMARK_GEMM_NT_STATIC(double, 31);
    BENCHMARK_GEMM_NT_STATIC(double, 32);
    BENCHMARK_GEMM_NT_STATIC(double, 33);
    BENCHMARK_GEMM_NT_STATIC(double, 34);
    BENCHMARK_GEMM_NT_STATIC(double, 35);
    BENCHMARK_GEMM_NT_STATIC(double, 36);
    BENCHMARK_GEMM_NT_STATIC(double, 37);
    BENCHMARK_GEMM_NT_STATIC(double, 38);
    BENCHMARK_GEMM_NT_STATIC(double, 39);
    BENCHMARK_GEMM_NT_STATIC(double, 40);
    BENCHMARK_GEMM_NT_STATIC(double, 41);
    BENCHMARK_GEMM_NT_STATIC(double, 42);
    BENCHMARK_GEMM_NT_STATIC(double, 43);
    BENCHMARK_GEMM_NT_STATIC(double, 44);
    BENCHMARK_GEMM_NT_STATIC(double, 45);
    BENCHMARK_GEMM_NT_STATIC(double, 46);
    BENCHMARK_GEMM_NT_STATIC(double, 47);
    BENCHMARK_GEMM_NT_STATIC(double, 48);
    BENCHMARK_GEMM_NT_STATIC(double, 49);
    BENCHMARK_GEMM_NT_STATIC(double, 50);
    BENCHMARK_GEMM_NT_STATIC(double, 300);
}
