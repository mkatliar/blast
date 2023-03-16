// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.



#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace blast :: benchmark
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
