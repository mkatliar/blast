// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blasfeo/Blasfeo.hpp>

#include <blazefeo/Blaze.hpp>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    template <typename T>
    void BM_potrf(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        size_t const n = m;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<T, blaze::columnMajor> blaze_A(m, m), blaze_L(m, m);
        makePositiveDefinite(blaze_A);
        
        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<T> blasfeo_A(blaze_A), blasfeo_L(m, m);
        
        // Do potrf with BLASFEO
        for (auto _ : state)
            potrf(m, blasfeo_A, 0, 0, blasfeo_L, 0, 0);

        // Calculated as \sum _{k=0}^{n-1} \sum _{j=0}^{k-1} \sum _{i=k}^{m-1} 2
        state.counters["flops"] = Counter((1 + 3 * m - 2 * n) * (n - 1) * n / 3, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    BENCHMARK_TEMPLATE(BM_potrf, double)->DenseRange(1, 300);
    BENCHMARK_TEMPLATE(BM_potrf, float)->DenseRange(1, 300);
}
