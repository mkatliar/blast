// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Gemm.hpp>
#include <test/Randomize.hpp>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    using namespace blaze;


    template <typename Real, size_t M, size_t N>
    static void BM_trmm_static_plain(State& state)
    {
        LowerMatrix<StaticMatrix<Real, M, M, columnMajor>> A;
        StaticMatrix<Real, M, N, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;

        randomize(A);
        randomize(B);

        for (auto _ : state)
        {
            C = trans(A) * B;
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(M * (M + 1) * N, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
    }


BENCHMARK_TEMPLATE(BM_trmm_static_plain, double, 20, 40);


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(n) BENCHMARK_TEMPLATE(BM_trmm_static_plain, double, n, n);
#include BOOST_PP_LOCAL_ITERATE()
}
