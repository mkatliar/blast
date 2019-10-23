#include <smoke/StaticPanelMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace smoke :: benchmark
{
    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_tn_impl(
        StaticPanelMatrix<T, K, M, P> const& A, StaticPanelMatrix<T, K, N, P> const& B, 
        StaticPanelMatrix<T, M, N, P> const& C, StaticPanelMatrix<T, M, N, P>& D);

    
    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_nn_impl(
        StaticPanelMatrix<T, M, K, P> const& A, StaticPanelMatrix<T, K, N, P> const& B, 
        StaticPanelMatrix<T, M, N, P> const& C, StaticPanelMatrix<T, M, N, P>& D);

    
    template <typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_nt_impl(
        StaticPanelMatrix<T, M, K, P> const& A, StaticPanelMatrix<T, N, K, P> const& B, 
        StaticPanelMatrix<T, M, N, P> const& C, StaticPanelMatrix<T, M, N, P>& D);


    template <size_t KM, size_t KN, typename Real, size_t M>
    static void BM_gemm_tn(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticPanelMatrix<Real, K, M> A;
        StaticPanelMatrix<Real, K, N> B;
        StaticPanelMatrix<Real, M, N> C;
        StaticPanelMatrix<Real, M, N> D;

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
            gemm_tn_impl<KM, KN>(A, B, C, D);

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    template <size_t KM, size_t KN, typename Real, size_t M>
    static void BM_gemm_nn(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticPanelMatrix<Real, K, M> A;
        StaticPanelMatrix<Real, K, N> B;
        StaticPanelMatrix<Real, M, N> C;
        StaticPanelMatrix<Real, M, N> D;

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
            gemm_nn_impl<KM, KN>(A, B, C, D);

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    template <typename Real, size_t M>
    static void BM_gemm_nt(::benchmark::State& state)
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
            gemm_nt_impl(A, B, C, D);

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }
    

    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 4);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 8);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 12);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 16);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 20);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 24);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 28);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 32);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 36);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 1, 1, double, 40);

    // BENCHMARK_TEMPLATE(BM_gemm_tn, 2, 1, double, 8);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 2, 1, double, 16);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 2, 1, double, 24);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 2, 1, double, 32);
    // BENCHMARK_TEMPLATE(BM_gemm_tn, 2, 1, double, 40);

    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 4);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 8);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 12);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 16);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 20);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 24);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 28);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 32);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 36);
    // BENCHMARK_TEMPLATE(BM_gemm_nn, 1, 1, double, 40);

    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 1);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 2);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 3);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 4);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 5);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 6);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 7);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 8);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 9);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 10);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 11);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 12);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 13);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 14);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 15);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 16);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 17);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 18);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 19);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 20);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 21);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 22);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 23);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 24);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 25);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 26);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 27);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 28);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 29);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 30);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 31);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 32);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 33);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 34);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 35);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 36);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 37);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 38);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 39);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 40);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 41);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 42);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 43);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 44);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 45);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 46);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 47);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 48);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 49);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double, 50);
}
