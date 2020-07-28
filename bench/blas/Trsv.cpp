#include <blazefeo/Blaze.hpp>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real, bool SO>
    static void BM_trsv(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        blaze::DynamicMatrix<Real, SO> A(M, M);        
        blaze::DynamicVector<Real, SO> B(M);
        blaze::DynamicVector<Real, SO> C(M);

        randomize(A);
        randomize(B);
        
        for (auto _ : state)
        {
            C = B;
            trsv(A, C, 'L', 'N', 'N');
        }
    }


    static void trsvBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Arg(1)->Arg(4)->Arg(35)->Arg(60);
    }


    BENCHMARK_TEMPLATE(BM_trsv, double, blaze::rowMajor)->Apply(trsvBenchArguments);
    BENCHMARK_TEMPLATE(BM_trsv, double, blaze::columnMajor)->Apply(trsvBenchArguments);
}
