// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Benchmark.hpp>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_trmm(::benchmark::State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(M, M);
        blaze::DynamicMatrix<Real, blaze::columnMajor> B(M, N);
        blaze::DynamicMatrix<Real, blaze::columnMajor> C(M, N);

        randomize(A);
        randomize(B);

        for (auto _ : state)
        {
            C = B;
            trmm(C, trans(std::as_const(A)), CblasLeft, CblasUpper, 1.);
        }
    }


    BENCHMARK_TEMPLATE(BM_trmm, double)->Apply(trmmBenchArguments);
}
