#include <blaze/Math.h>

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
        
        blaze::StaticMatrix<Real, K, M, blaze::columnMajor> A;
        randomize(A);

        blaze::StaticMatrix<Real, K, N, blaze::columnMajor> B;
        randomize(B);

        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> C;
        randomize(C);
        
        for (auto _ : state)
        {
            C += trans(A) * B;
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

        blaze::DynamicMatrix<Real, blaze::columnMajor> A(m, m);
        randomize(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(m, m);
        randomize(B);

        blaze::DynamicMatrix<Real, blaze::columnMajor> C(m, m);
        randomize(C);
        
        for (auto _ : state)
        {
            C += trans(A) * B;
            ::benchmark::DoNotOptimize(A);
            ::benchmark::DoNotOptimize(B);
            ::benchmark::DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(2 * m * m * m, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    template <typename Real>
    static void BM_gemm_loop_naive(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        std::vector<Real> A(m * m), B(m * m), C(m * m), D(m * m);
        double * pA = A.data(), * pB = B.data(), * pC = C.data(), * pD = D.data();
        
        for (auto _ : state)
        {
            for (size_t j = 0; j < m; ++j)
                for (size_t i = 0; i < m; ++i)
                {
                    double s = pC[i + j * m];

                    for (size_t k = 0; k < m; ++k)
                        s += pA[k + i * m] * pB[k + j * m];

                    pD[i + j * m] = s;
                }
        }

        state.counters["flops"] = Counter(2 * m * m * m, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    template <typename Real>
    static void BM_gemm_loop_optimized(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        std::vector<Real> A(m * m), B(m * m), C(m * m), D(m * m);
        double const * pA = A.data();
        double const * pB = B.data();
        double const * pC = C.data();
        double * pD = D.data();
        
        for (auto _ : state)
        {
            for (size_t j = 0; j < m; ++j)
                for (size_t i = 0; i < m; ++i)
                {
                    double s = pC[i + j * m];

                    double const * pa = pA + i * m;
                    double const * pb = pB + j * m;

                    for (size_t k = 0; k < m; ++k)
                        s += pa[k] * pb[k];

                    pD[i + j * m] = s;
                }
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
    BENCHMARK_GEMM_STATIC_100(1);
    BENCHMARK_GEMM_STATIC_100(2);
    BENCHMARK_GEMM_STATIC(300);

    BENCHMARK_TEMPLATE(BM_gemm_dynamic, double)->DenseRange(1, 50);
    
    BENCHMARK_TEMPLATE(BM_gemm_loop_naive, double)->DenseRange(1, 50);
    BENCHMARK_TEMPLATE(BM_gemm_loop_optimized, double)->DenseRange(1, 50);
}
