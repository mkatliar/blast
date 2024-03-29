// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace blast :: benchmark
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
