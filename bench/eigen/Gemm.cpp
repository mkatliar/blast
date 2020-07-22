#include <Eigen/Dense>

#include <benchmark/benchmark.h>

#include <vector>


#define BENCHMARK_GEMM_STATIC(N) \
    BENCHMARK_TEMPLATE(BM_gemm_static, double, N); \
    BENCHMARK_TEMPLATE(BM_gemm_static, float, N);

#define BENCHMARK_GEMM_STATIC_10(tens) \
    BENCHMARK_GEMM_STATIC(tens##0); \
    BENCHMARK_GEMM_STATIC(tens##1); \
    BENCHMARK_GEMM_STATIC(tens##2); \
    BENCHMARK_GEMM_STATIC(tens##3); \
    BENCHMARK_GEMM_STATIC(tens##4); \
    BENCHMARK_GEMM_STATIC(tens##5); \
    BENCHMARK_GEMM_STATIC(tens##6); \
    BENCHMARK_GEMM_STATIC(tens##7); \
    BENCHMARK_GEMM_STATIC(tens##8); \
    BENCHMARK_GEMM_STATIC(tens##9);

#define BENCHMARK_GEMM_STATIC_100(hundreds) \
    BENCHMARK_GEMM_STATIC_10(hundreds##0); \
    BENCHMARK_GEMM_STATIC_10(hundreds##1); \
    BENCHMARK_GEMM_STATIC_10(hundreds##2); \
    BENCHMARK_GEMM_STATIC_10(hundreds##3); \
    BENCHMARK_GEMM_STATIC_10(hundreds##4); \
    BENCHMARK_GEMM_STATIC_10(hundreds##5); \
    BENCHMARK_GEMM_STATIC_10(hundreds##6); \
    BENCHMARK_GEMM_STATIC_10(hundreds##7); \
    BENCHMARK_GEMM_STATIC_10(hundreds##8); \
    BENCHMARK_GEMM_STATIC_10(hundreds##9);


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    template <typename Real, size_t M>
    static void BM_gemm_static(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;
        
        Eigen::Matrix<Real, K, M, Eigen::ColMajor> A;
        A.setRandom();

        Eigen::Matrix<Real, K, N, Eigen::ColMajor> B;
        B.setRandom();

        Eigen::Matrix<Real, M, N, Eigen::ColMajor> C;
        C.setRandom();
        
        for (auto _ : state)
        {
            C += A.transpose() * B;
            ::benchmark::DoNotOptimize(A);
            ::benchmark::DoNotOptimize(B);
            ::benchmark::DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
        state.counters["k"] = K;
    }


    template <typename Real>
    static void BM_gemm_dynamic(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A(m, m);
        A.setRandom();

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> B(m, m);
        B.setRandom();

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> C(m, m);
        C.setRandom();
        
        for (auto _ : state)
        {
            C += A.transpose() * B;
            ::benchmark::DoNotOptimize(A);
            ::benchmark::DoNotOptimize(B);
            ::benchmark::DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(2 * m * m * m, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    BENCHMARK_GEMM_STATIC(1);
    BENCHMARK_GEMM_STATIC(2);
    BENCHMARK_GEMM_STATIC(3);
    BENCHMARK_GEMM_STATIC(4);
    BENCHMARK_GEMM_STATIC(5);
    BENCHMARK_GEMM_STATIC(6);
    BENCHMARK_GEMM_STATIC(7);
    BENCHMARK_GEMM_STATIC(8);
    BENCHMARK_GEMM_STATIC(9);
    BENCHMARK_GEMM_STATIC_10(1);
    BENCHMARK_GEMM_STATIC_10(2);
    BENCHMARK_GEMM_STATIC_10(3);
    BENCHMARK_GEMM_STATIC_10(4);
    BENCHMARK_GEMM_STATIC_10(5);
    BENCHMARK_GEMM_STATIC_10(6);
    BENCHMARK_GEMM_STATIC_10(7);
    BENCHMARK_GEMM_STATIC_10(8);
    BENCHMARK_GEMM_STATIC_10(9);
    // BENCHMARK_GEMM_STATIC_100(1);
    // BENCHMARK_GEMM_STATIC_100(2);
    // BENCHMARK_GEMM_STATIC(300);

    BENCHMARK_TEMPLATE(BM_gemm_dynamic, double)->DenseRange(1, 50);
}
