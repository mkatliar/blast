// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/algorithm/Gemm.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/blaze/Math.hpp>

#include <bench/Gemm.hpp>

#include <test/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm_static_plain(State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        StaticMatrix<Real, M, K, columnMajor> A;
        StaticMatrix<Real, K, N, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;
        StaticMatrix<Real, M, N, columnMajor> D;
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


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(n) BENCHMARK_TEMPLATE(BM_gemm_static_plain, double, n);
#include BOOST_PP_LOCAL_ITERATE()
}
