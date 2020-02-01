#include <Eigen/Dense>

#include <benchmark/benchmark.h>

#include <vector>


#define BENCHMARK_GEMM_STATIC(N) \
    BENCHMARK_TEMPLATE(BM_gemm_static, double, N); \
    BENCHMARK_TEMPLATE(BM_gemm_static, float, N);


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
    BENCHMARK_GEMM_STATIC(10);
    BENCHMARK_GEMM_STATIC(11);
    BENCHMARK_GEMM_STATIC(12);
    BENCHMARK_GEMM_STATIC(13);
    BENCHMARK_GEMM_STATIC(14);
    BENCHMARK_GEMM_STATIC(15);
    BENCHMARK_GEMM_STATIC(16);
    BENCHMARK_GEMM_STATIC(17);
    BENCHMARK_GEMM_STATIC(18);
    BENCHMARK_GEMM_STATIC(19);
    BENCHMARK_GEMM_STATIC(20);
    BENCHMARK_GEMM_STATIC(21);
    BENCHMARK_GEMM_STATIC(22);
    BENCHMARK_GEMM_STATIC(23);
    BENCHMARK_GEMM_STATIC(24);
    BENCHMARK_GEMM_STATIC(25);
    BENCHMARK_GEMM_STATIC(26);
    BENCHMARK_GEMM_STATIC(27);
    BENCHMARK_GEMM_STATIC(28);
    BENCHMARK_GEMM_STATIC(29);
    BENCHMARK_GEMM_STATIC(30);
    BENCHMARK_GEMM_STATIC(31);
    BENCHMARK_GEMM_STATIC(32);
    BENCHMARK_GEMM_STATIC(33);
    BENCHMARK_GEMM_STATIC(34);
    BENCHMARK_GEMM_STATIC(35);
    BENCHMARK_GEMM_STATIC(36);
    BENCHMARK_GEMM_STATIC(37);
    BENCHMARK_GEMM_STATIC(38);
    BENCHMARK_GEMM_STATIC(39);
    BENCHMARK_GEMM_STATIC(40);
    BENCHMARK_GEMM_STATIC(41);
    BENCHMARK_GEMM_STATIC(42);
    BENCHMARK_GEMM_STATIC(43);
    BENCHMARK_GEMM_STATIC(44);
    BENCHMARK_GEMM_STATIC(45);
    BENCHMARK_GEMM_STATIC(46);
    BENCHMARK_GEMM_STATIC(47);
    BENCHMARK_GEMM_STATIC(48);
    BENCHMARK_GEMM_STATIC(49);
    BENCHMARK_GEMM_STATIC(50);

    BENCHMARK_TEMPLATE(BM_gemm_dynamic, double)->DenseRange(1, 50);
}
