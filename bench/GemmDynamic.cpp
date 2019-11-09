#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    template <typename MT1, bool SO1, typename MT2, bool SO2, typename MT3, bool SO3, typename MT4, bool SO4>
    __attribute((noinline)) static void gemm_nt_noinline(
        Matrix<MT1, SO1> const& A, Matrix<MT2, SO2> const& B, 
        Matrix<MT3, SO3> const& C, Matrix<MT4, SO4>& D)
    {
        gemm_nt(~A, ~B, ~C, ~D);
    }


    template <typename Real>
    static void BM_gemm_nt_dynamic_inline(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = M;
        size_t const K = M;

        DynamicPanelMatrix<Real> A(M, K);
        DynamicPanelMatrix<Real> B(N, K);
        DynamicPanelMatrix<Real> C(M, N);
        DynamicPanelMatrix<Real> D(M, N);

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


    template <typename Real>
    static void BM_gemm_nt_dynamic_noinline(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = M;
        size_t const K = M;

        DynamicPanelMatrix<Real> A(M, K);
        DynamicPanelMatrix<Real> B(N, K);
        DynamicPanelMatrix<Real> C(M, N);
        DynamicPanelMatrix<Real> D(M, N);

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
    

    BENCHMARK_TEMPLATE(BM_gemm_nt_dynamic_inline, double)->DenseRange(1, 300);
    BENCHMARK_TEMPLATE(BM_gemm_nt_dynamic_noinline, double)->DenseRange(1, 300);
}
