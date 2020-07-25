#include <blasfeo/Blasfeo.hpp>

#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blasfeo :: benchmark
{
    void BM_syrk(::benchmark::State& state)
    {
        size_t const m = state.range(0), k = state.range(1);

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(m, k);
        blaze::LowerMatrix<blaze::DynamicMatrix<double, blaze::columnMajor>> blaze_C(m, m);
        randomize(blaze_A);
        randomize(blaze_C);
        
        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> A(blaze_A), C(blaze_C), D(m, m);
        
        // Do syrk-potrf with BLASFEO
        for (auto _ : state)
            syrk_ln(m, k, 1., A, 0, 0, A, 0, 0, 1., C, 0, 0, D, 0, 0);
    }


    BENCHMARK(BM_syrk)->Args({5, 4})->Args({40, 20})->Args({60, 30});
}
