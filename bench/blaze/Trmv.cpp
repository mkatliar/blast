// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_trmv_dynamic(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        blaze::LowerMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> A(M, M);
        blaze::DynamicVector<Real, blaze::columnVector> B(M);
        blaze::DynamicVector<Real, blaze::columnVector> C(M);

        randomize(A);
        randomize(B);

        for (auto _ : state)
            ::benchmark::DoNotOptimize(C = trans(A) * B);
    }


    template <typename Real, size_t M>
    static void BM_trmv_Blaze_Static(::benchmark::State& state)
    {
        blaze::LowerMatrix<blaze::StaticMatrix<Real, M, M, blaze::columnMajor>> A;
        blaze::StaticVector<Real, M, blaze::columnVector> B;
        blaze::StaticVector<Real, M, blaze::columnVector> C;

        randomize(A);
        randomize(B);

        for (auto _ : state)
            ::benchmark::DoNotOptimize(C = trans(A) * B);
    }


    static void trmvBenchArguments(::benchmark::internal::Benchmark* b)
    {
        b->Arg(1)->Arg(4)->Arg(35);
    }


    BENCHMARK_TEMPLATE(BM_trmv_dynamic, double)->Apply(trmvBenchArguments);
    BENCHMARK_TEMPLATE(BM_trmv_Blaze_Static, double, 1);
    BENCHMARK_TEMPLATE(BM_trmv_Blaze_Static, double, 4);
    BENCHMARK_TEMPLATE(BM_trmv_Blaze_Static, double, 35);
}
