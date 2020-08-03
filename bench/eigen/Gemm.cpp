#include <Eigen/Dense>

#include <bench/Benchmark.hpp>

#include <vector>


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


    BENCHMARK_TEMPLATE(BM_gemm_dynamic, double)->DenseRange(1, 50);


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(N) \
    BENCHMARK_TEMPLATE(BM_gemm_static, double, N); \
    BENCHMARK_TEMPLATE(BM_gemm_static, float, N);
#include BOOST_PP_LOCAL_ITERATE()
}
