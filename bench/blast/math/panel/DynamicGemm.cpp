// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/panel/Gemm.hpp>

#include <bench/Gemm.hpp>

#include <test/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_gemm_dynamic_panel(State& state)
    {
        size_t const M = state.range(0);
        size_t const N = M;
        size_t const K = M;

        DynamicPanelMatrix<Real, columnMajor> A(M, K);
        DynamicPanelMatrix<Real, columnMajor> B(N, K);
        DynamicPanelMatrix<Real, columnMajor> C(M, N);
        DynamicPanelMatrix<Real, columnMajor> D(M, N);
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


    BENCHMARK_TEMPLATE(BM_gemm_dynamic_panel, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
    BENCHMARK_TEMPLATE(BM_gemm_dynamic_panel, float)->DenseRange(1, BENCHMARK_MAX_GEMM);
}
