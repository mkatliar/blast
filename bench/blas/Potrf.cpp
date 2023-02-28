// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Complexity.hpp>

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
        blaze::DynamicMatrix<T, blaze::columnMajor> A(m, m), L(m, m);
        makePositiveDefinite(A);

        // Do potrf
        for (auto _ : state)
        {
            L = A;

            int info;
            blaze::potrf('L', m, data(L), spacing(L), &info);
        }

        setCounters(state.counters, complexityPotrf(m, n));
        state.counters["m"] = m;
    }


    BENCHMARK_TEMPLATE(BM_potrf, double)->DenseRange(1, 300);
    BENCHMARK_TEMPLATE(BM_potrf, float)->DenseRange(1, 300);
}
