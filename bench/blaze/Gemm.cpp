#include <blazefeo/Blaze.hpp>

#include <bench/Gemm.hpp>

#include <vector>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm_static(State& state)
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
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
        state.counters["k"] = K;
    }


    template <typename Real>
    static void BM_gemm_dynamic(State& state)
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
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(2 * m * m * m, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    template <typename Real>
    static void BM_gemm_loop_naive(State& state)
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
    static void BM_gemm_loop_optimized(State& state)
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


    BENCHMARK_TEMPLATE(BM_gemm_dynamic, double)->DenseRange(1, BENCHMARK_MAX_GEMM);    
    BENCHMARK_TEMPLATE(BM_gemm_loop_naive, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
    BENCHMARK_TEMPLATE(BM_gemm_loop_optimized, double)->DenseRange(1, BENCHMARK_MAX_GEMM);


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(N) \
    BENCHMARK_TEMPLATE(BM_gemm_static, double, N); \
    BENCHMARK_TEMPLATE(BM_gemm_static, float, N);
#include BOOST_PP_LOCAL_ITERATE()
}
