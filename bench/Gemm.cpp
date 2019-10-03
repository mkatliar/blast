#include <smoke/StaticMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <random>
#include <memory>


namespace smoke :: benchmark
{
    template <typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_tn_impl(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D);

    
    template <typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_nn_impl(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D);


    template <typename T, size_t M, size_t N, size_t P>
    static void randomize(StaticMatrix<T, M, N, P>& A)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-1.0, 1.0);	

        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = 0; j < A.columns(); ++j)
                A(i, j) = dis(gen);
    }


    template <typename Real, size_t M>
    static void BM_gemm_tn(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticMatrix<Real, K, M> A;
        StaticMatrix<Real, K, N> B;
        StaticMatrix<Real, M, N> C;
        StaticMatrix<Real, M, N> D;

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
            gemm_tn_impl(A, B, C, D);

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    template <typename Real, size_t M>
    static void BM_gemm_nn(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticMatrix<Real, K, M> A;
        StaticMatrix<Real, K, N> B;
        StaticMatrix<Real, M, N> C;
        StaticMatrix<Real, M, N> D;

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
            gemm_nn_impl(A, B, C, D);

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }
    

    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 4);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 8);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 12);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 16);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 20);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 24);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 28);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 32);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 36);
    BENCHMARK_TEMPLATE(BM_gemm_tn, double, 40);

    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 4);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 8);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 12);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 16);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 20);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 24);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 28);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 32);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 36);
    BENCHMARK_TEMPLATE(BM_gemm_nn, double, 40);
}
