#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_syrk_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(M, N);        
        blaze::DynamicMatrix<Real, blaze::columnMajor> B(N, N);

        randomize(A);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = trans(A) * A);
    }


    template <typename Real>
    static void BM_syrk_symmetric_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(M, N);        
        blaze::SymmetricMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> B(N);

        randomize(A);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = declsym(trans(A) * A));
    }


    template <typename Real, size_t M, size_t N>
    static void BM_syrk_static(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> A;        
        blaze::StaticMatrix<Real, N, N, blaze::columnMajor> B;

        randomize(A);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = trans(A) * A);
    }


    template <typename Real, size_t M, size_t N>
    static void BM_syrk_declsym_static(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> A;        
        blaze::StaticMatrix<Real, N, N, blaze::columnMajor> B;

        randomize(A);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = declsym(trans(A) * A));
    }


    template <typename Real, size_t M, size_t N>
    static void BM_syrk_symmetric_declsym_static(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> A;
        blaze::SymmetricMatrix<blaze::StaticMatrix<Real, N, N, blaze::columnMajor>> B;

        randomize(A);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = declsym(trans(A) * A));
    }


    template <typename Real, size_t M, size_t N, bool SO1, bool SO2>
    void syrkStaticLoop(blaze::StaticMatrix<Real, M, N, SO1> const& A, blaze::StaticMatrix<Real, N, N, SO2>& B)
    {
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t i = j; i < N; ++i)
                B(i, j) = dot(column(A, i), column(A, j));
        }
    }


    template <typename Real, size_t M, size_t N>
    static void BM_syrk_loop_static(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> A;        
        blaze::StaticMatrix<Real, N, N, blaze::columnMajor> B;

        randomize(A);
        
        for (auto _ : state)
        {
            syrkStaticLoop(A, B);
        }
    }


    static void syrkBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Args({4, 5})->Args({30, 35});
    }


    BENCHMARK_TEMPLATE(BM_syrk_dynamic, double)->Apply(syrkBenchArguments);
    BENCHMARK_TEMPLATE(BM_syrk_symmetric_dynamic, double)->Apply(syrkBenchArguments);
    BENCHMARK_TEMPLATE(BM_syrk_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_static, double, 30, 35);
    BENCHMARK_TEMPLATE(BM_syrk_declsym_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_declsym_static, double, 30, 35);
    BENCHMARK_TEMPLATE(BM_syrk_symmetric_declsym_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_symmetric_declsym_static, double, 30, 35);

    BENCHMARK_TEMPLATE(BM_syrk_loop_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_loop_static, double, 30, 35);
}
