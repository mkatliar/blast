#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_trsv_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        blaze::LowerMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> A(M, M);        
        blaze::DynamicVector<Real, blaze::columnVector> B(M);
        blaze::DynamicVector<Real, blaze::columnVector> C(M);

        randomize(A);
        randomize(B);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(C = inv(A) * B);
    }


    template <typename Real, size_t M>
    static void BM_trsv_Blaze_Static(::benchmark::State& state)
    {
        blaze::LowerMatrix<blaze::StaticMatrix<Real, M, M, blaze::columnMajor>> A;        
        blaze::StaticVector<Real, M, blaze::columnVector> B;
        blaze::StaticVector<Real, M, blaze::columnVector> C;

        randomize(A);
        randomize(B);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(C = inv(A) * B);
    }


    static void trsvBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Arg(1)->Arg(4)->Arg(35)->Arg(60);
    }


    BENCHMARK_TEMPLATE(BM_trsv_dynamic, double)->Apply(trsvBenchArguments);
    
    BENCHMARK_TEMPLATE(BM_trsv_Blaze_Static, double, 1);
    BENCHMARK_TEMPLATE(BM_trsv_Blaze_Static, double, 4);
    BENCHMARK_TEMPLATE(BM_trsv_Blaze_Static, double, 35);
    BENCHMARK_TEMPLATE(BM_trsv_Blaze_Static, double, 60);
}
