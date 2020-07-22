#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_GEMM_NT_STATIC(size) \
    BENCHMARK_TEMPLATE(BM_gemm_nt_static_inline, double, size); \
    BENCHMARK_TEMPLATE(BM_gemm_nt_static_noinline, double, size); \
    BENCHMARK_TEMPLATE(BM_gemm_nt_static_inline, float, size); \
    BENCHMARK_TEMPLATE(BM_gemm_nt_static_noinline, float, size);


#define BENCHMARK_GEMM_NT_STATIC_10(tens) \
    BENCHMARK_GEMM_NT_STATIC(tens##0); \
    BENCHMARK_GEMM_NT_STATIC(tens##1); \
    BENCHMARK_GEMM_NT_STATIC(tens##2); \
    BENCHMARK_GEMM_NT_STATIC(tens##3); \
    BENCHMARK_GEMM_NT_STATIC(tens##4); \
    BENCHMARK_GEMM_NT_STATIC(tens##5); \
    BENCHMARK_GEMM_NT_STATIC(tens##6); \
    BENCHMARK_GEMM_NT_STATIC(tens##7); \
    BENCHMARK_GEMM_NT_STATIC(tens##8); \
    BENCHMARK_GEMM_NT_STATIC(tens##9);

#define BENCHMARK_GEMM_NT_STATIC_100(hundreds) \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##0); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##1); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##2); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##3); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##4); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##5); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##6); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##7); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##8); \
    BENCHMARK_GEMM_NT_STATIC_10(hundreds##9);


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


    BENCHMARK_GEMM_NT_STATIC(1);
    BENCHMARK_GEMM_NT_STATIC(2);
    BENCHMARK_GEMM_NT_STATIC(3);
    BENCHMARK_GEMM_NT_STATIC(4);
    BENCHMARK_GEMM_NT_STATIC(5);
    BENCHMARK_GEMM_NT_STATIC(6);
    BENCHMARK_GEMM_NT_STATIC(7);
    BENCHMARK_GEMM_NT_STATIC(8);
    BENCHMARK_GEMM_NT_STATIC(9);
    BENCHMARK_GEMM_NT_STATIC_10(1);
    BENCHMARK_GEMM_NT_STATIC_10(2);
    BENCHMARK_GEMM_NT_STATIC_10(3);
    BENCHMARK_GEMM_NT_STATIC_10(4);
    BENCHMARK_GEMM_NT_STATIC_10(5);
    BENCHMARK_GEMM_NT_STATIC_10(6);
    BENCHMARK_GEMM_NT_STATIC_10(7);
    BENCHMARK_GEMM_NT_STATIC_10(8);
    BENCHMARK_GEMM_NT_STATIC_10(9);
    BENCHMARK_GEMM_NT_STATIC_100(1);
    BENCHMARK_GEMM_NT_STATIC_100(2);
    BENCHMARK_GEMM_NT_STATIC(300);
}
