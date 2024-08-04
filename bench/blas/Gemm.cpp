// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Gemm.hpp>

#include <test/Randomize.hpp>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    using namespace ::benchmark;


    template <typename Real>
    static void BM_gemm(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        blaze::DynamicMatrix<Real, blaze::columnMajor> A(m, m);
        randomize(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(m, m);
        randomize(B);

        blaze::DynamicMatrix<Real, blaze::columnMajor> C(m, m);
        randomize(C);

        Real alpha, beta;
        randomize(alpha);
        randomize(beta);

        for (auto _ : state)
            gemm(C, trans(A), B, alpha, beta);

        setCounters(state.counters, complexityGemm(m, m, m));
        state.counters["m"] = m;
    }

    BENCHMARK_TEMPLATE(BM_gemm, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
    BENCHMARK_TEMPLATE(BM_gemm, float)->DenseRange(1, BENCHMARK_MAX_GEMM);
}
