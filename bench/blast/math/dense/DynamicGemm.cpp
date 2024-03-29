// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/dense/Gemm.hpp>

#include <blaze/math/DynamicMatrix.h>

#include <bench/Gemm.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


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

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
        {
            gemm(1., A, trans(B), 1., C, D);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_gemm_dynamic_plain, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
}
