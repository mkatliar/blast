#include <blazefeo/Blaze.hpp>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t N, bool SO>
    static void BM_sqrNorm(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, N, N, SO> A;
        randomize(A);
        
        for (auto _ : state)
        {
            for (size_t k = 0; k < N; ++k)
            {
                auto const A10 = submatrix(A, k, 0, 1, k);
                ::benchmark::DoNotOptimize(sqrNorm(A10));
            }
        }
    }


    template <typename Real, size_t N, bool SO>
    static void BM_dotProductWithSelf(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, N, N, SO> A;
        randomize(A);
        
        for (auto _ : state)
        {
            for (size_t k = 0; k < N; ++k)
            {
                auto const A10 = submatrix(A, k, 0, 1, k);
                Real x {};
                ::benchmark::DoNotOptimize(x = dot(row(A10, 0), conj(row(A10, 0))));
            }
        }
    }


    BENCHMARK_TEMPLATE(BM_sqrNorm, double, 35, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_sqrNorm, double, 35, blaze::rowMajor);
    
    BENCHMARK_TEMPLATE(BM_dotProductWithSelf, double, 35, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_dotProductWithSelf, double, 35, blaze::rowMajor);
}
