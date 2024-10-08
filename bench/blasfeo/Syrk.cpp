// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blasfeo/Blasfeo.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Syrk.hpp>


namespace blast :: benchmark
{
    template <typename Real>
    void BM_syrk(State& state)
    {
        size_t const m = state.range(0), k = state.range(1);

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<Real, blaze::columnMajor> blaze_A(m, k);
        blaze::LowerMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> blaze_C(m, m);
        randomize(blaze_A);
        randomize(blaze_C);

        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<Real> A(blaze_A), C(blaze_C), D(m, m);

        // Do syrk-potrf with BLASFEO
        for (auto _ : state)
            syrk_ln(m, k, 1., A, 0, 0, A, 0, 0, 1., C, 0, 0, D, 0, 0);

        setCounters(state.counters, complexitySyrk(m, k));
        state.counters["m"] = m;
    }


    // BENCHMARK_TEMPLATE(BM_syrk, double)->Args({5, 4})->Args({40, 20})->Args({60, 30});
    BENCHMARK_TEMPLATE(BM_syrk, double)->Apply(syrkBenchArguments);
}
