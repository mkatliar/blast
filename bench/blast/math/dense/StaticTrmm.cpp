// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/dense/Trmm.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/blaze/Math.hpp>

#include <bench/Gemm.hpp>
#include <test/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename Real, size_t M, size_t N>
    static void BM_trmm_left_upper_static_plain(State& state)
    {
        StaticMatrix<Real, M, M, columnMajor> A;
        StaticMatrix<Real, M, N, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;

        randomize(A);
        randomize(B);

        for (auto _ : state)
        {
            trmmLeftUpper(1., A, B, C);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(M * (M + 1) * N, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
    }


    template <typename Real, size_t M, size_t N>
    static void BM_trmm_right_lower_static_plain(State& state)
    {
        StaticMatrix<Real, N, N, columnMajor> A;
        StaticMatrix<Real, M, N, columnMajor> B;
        StaticMatrix<Real, M, N, columnMajor> C;

        randomize(A);
        randomize(B);

        for (auto _ : state)
        {
            trmmRightLower(1., B, A, C);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(N * (N + 1) * M, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
    }


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_TRMM)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_trmm_left_upper_static_plain, double, n, n); \
    BENCHMARK_TEMPLATE(BM_trmm_right_lower_static_plain, double, n, n);
#include BOOST_PP_LOCAL_ITERATE()
}
