// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/dense/Gemm.hpp>

#include <blaze/math/StaticMatrix.h>

#include <bench/Gemm.hpp>
#include <test/Randomize.hpp>

#include <random>
#include <memory>


namespace blast :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm_static_plain(State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticMatrix<Real, M, K, columnMajor> A;
        StaticMatrix<Real, N, K, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;
        StaticMatrix<Real, M, N, columnMajor> D;

        randomize(A);
        randomize(B);
        randomize(C);

        for (auto _ : state)
        {
            // gemm(1., A, trans(B), 1., C, D);
            gemm(A, B, C, D);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
    }


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(n) BENCHMARK_TEMPLATE(BM_gemm_static_plain, double, n);
#include BOOST_PP_LOCAL_ITERATE()
}
