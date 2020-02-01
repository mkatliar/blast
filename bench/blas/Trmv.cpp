#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_trmv(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(M, M);        
        blaze::DynamicVector<Real, blaze::columnVector> B(M);
        blaze::DynamicVector<Real, blaze::columnVector> C(M);

        randomize(A);
        randomize(B);
        
        for (auto _ : state)
        {
            C = B;
            trmv(C, trans(std::as_const(A)), CblasUpper);
        }
    }


    static void trmvBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Arg(1)->Arg(4)->Arg(35);
    }



    BENCHMARK_TEMPLATE(BM_trmv, double)->Apply(trmvBenchArguments);
}
