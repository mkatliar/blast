#include <blazefeo/math/dense/Gemm.hpp>

#include <blaze/math/StaticMatrix.h>

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


#define BENCHMARK_GEMM2_NT_STATIC(size) \
    BENCHMARK_TEMPLATE(BM_gemm2_nt_static, double, size);


#define BENCHMARK_GEMM2_NT_STATIC_10(tens) \
    BENCHMARK_GEMM2_NT_STATIC(tens##0); \
    BENCHMARK_GEMM2_NT_STATIC(tens##1); \
    BENCHMARK_GEMM2_NT_STATIC(tens##2); \
    BENCHMARK_GEMM2_NT_STATIC(tens##3); \
    BENCHMARK_GEMM2_NT_STATIC(tens##4); \
    BENCHMARK_GEMM2_NT_STATIC(tens##5); \
    BENCHMARK_GEMM2_NT_STATIC(tens##6); \
    BENCHMARK_GEMM2_NT_STATIC(tens##7); \
    BENCHMARK_GEMM2_NT_STATIC(tens##8); \
    BENCHMARK_GEMM2_NT_STATIC(tens##9);

#define BENCHMARK_GEMM2_NT_STATIC_100(hundreds) \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##0); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##1); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##2); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##3); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##4); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##5); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##6); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##7); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##8); \
    BENCHMARK_GEMM2_NT_STATIC_10(hundreds##9);


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm2_nt_static(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticMatrix<Real, M, K, columnMajor> A;
        StaticMatrix<Real, N, K, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;
        StaticMatrix<Real, M, N, columnMajor> D;

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
        {
            gemm(1., A, trans(B), 1., C, D);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    BENCHMARK_GEMM2_NT_STATIC(1);
    BENCHMARK_GEMM2_NT_STATIC(2);
    BENCHMARK_GEMM2_NT_STATIC(3);
    BENCHMARK_GEMM2_NT_STATIC(4);
    BENCHMARK_GEMM2_NT_STATIC(5);
    BENCHMARK_GEMM2_NT_STATIC(6);
    BENCHMARK_GEMM2_NT_STATIC(7);
    BENCHMARK_GEMM2_NT_STATIC(8);
    BENCHMARK_GEMM2_NT_STATIC(9);
    BENCHMARK_GEMM2_NT_STATIC_10(1);
    BENCHMARK_GEMM2_NT_STATIC_10(2);
    BENCHMARK_GEMM2_NT_STATIC_10(3);
    BENCHMARK_GEMM2_NT_STATIC_10(4);
    BENCHMARK_GEMM2_NT_STATIC_10(5);
    BENCHMARK_GEMM2_NT_STATIC_10(6);
    BENCHMARK_GEMM2_NT_STATIC_10(7);
    BENCHMARK_GEMM2_NT_STATIC_10(8);
    BENCHMARK_GEMM2_NT_STATIC_10(9);
    BENCHMARK_GEMM2_NT_STATIC_100(1);
    BENCHMARK_GEMM2_NT_STATIC_100(2);
    BENCHMARK_GEMM2_NT_STATIC(300);
}
