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
    BENCHMARK_GEMM_NT_STATIC(10);
    BENCHMARK_GEMM_NT_STATIC(11);
    BENCHMARK_GEMM_NT_STATIC(12);
    BENCHMARK_GEMM_NT_STATIC(13);
    BENCHMARK_GEMM_NT_STATIC(14);
    BENCHMARK_GEMM_NT_STATIC(15);
    BENCHMARK_GEMM_NT_STATIC(16);
    BENCHMARK_GEMM_NT_STATIC(17);
    BENCHMARK_GEMM_NT_STATIC(18);
    BENCHMARK_GEMM_NT_STATIC(19);
    BENCHMARK_GEMM_NT_STATIC(20);
    BENCHMARK_GEMM_NT_STATIC(21);
    BENCHMARK_GEMM_NT_STATIC(22);
    BENCHMARK_GEMM_NT_STATIC(23);
    BENCHMARK_GEMM_NT_STATIC(24);
    BENCHMARK_GEMM_NT_STATIC(25);
    BENCHMARK_GEMM_NT_STATIC(26);
    BENCHMARK_GEMM_NT_STATIC(27);
    BENCHMARK_GEMM_NT_STATIC(28);
    BENCHMARK_GEMM_NT_STATIC(29);
    BENCHMARK_GEMM_NT_STATIC(30);
    BENCHMARK_GEMM_NT_STATIC(31);
    BENCHMARK_GEMM_NT_STATIC(32);
    BENCHMARK_GEMM_NT_STATIC(33);
    BENCHMARK_GEMM_NT_STATIC(34);
    BENCHMARK_GEMM_NT_STATIC(35);
    BENCHMARK_GEMM_NT_STATIC(36);
    BENCHMARK_GEMM_NT_STATIC(37);
    BENCHMARK_GEMM_NT_STATIC(38);
    BENCHMARK_GEMM_NT_STATIC(39);
    BENCHMARK_GEMM_NT_STATIC(40);
    BENCHMARK_GEMM_NT_STATIC(41);
    BENCHMARK_GEMM_NT_STATIC(42);
    BENCHMARK_GEMM_NT_STATIC(43);
    BENCHMARK_GEMM_NT_STATIC(44);
    BENCHMARK_GEMM_NT_STATIC(45);
    BENCHMARK_GEMM_NT_STATIC(46);
    BENCHMARK_GEMM_NT_STATIC(47);
    BENCHMARK_GEMM_NT_STATIC(48);
    BENCHMARK_GEMM_NT_STATIC(49);
    BENCHMARK_GEMM_NT_STATIC(50);
}
