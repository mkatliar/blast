// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/algorithm/Gemm.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/blaze/Math.hpp>

#include <bench/Gemm.hpp>

#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_gemm_dynamic_plain(State& state)
    {
        size_t const M = state.range(0);
        size_t const N = M;
        size_t const K = M;

        DynamicMatrix<Real, columnMajor> A(M, K);
        DynamicMatrix<Real, columnMajor> B(N, K);
        DynamicMatrix<Real, columnMajor> C(M, N);
        DynamicMatrix<Real, columnMajor> D(M, N);
        Real alpha, beta;

        randomize(A);
        randomize(B);
        randomize(C);
        randomize(alpha);
        randomize(beta);

        for (auto _ : state)
        {
            gemm(alpha, A, trans(B), beta, C, D);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        setCounters(state.counters, complexityGemm(M, N, K));
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_gemm_dynamic_plain, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
}
