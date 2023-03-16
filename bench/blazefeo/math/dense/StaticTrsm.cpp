// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "blazefeo/math/UpLo.hpp"
#include <blazefeo/math/dense/Trsm.hpp>

#include <blaze/math/StaticMatrix.h>

#include <bench/Trsm.hpp>


namespace blazefeo :: benchmark
{
    template <typename Real, UpLo UPLO, bool UNIT, size_t M, size_t N, bool SO>
    static void BM_trsm_left_dense_static(State& state)
    {
        StaticMatrix<Real, M, M, SO> A;
        StaticMatrix<Real, M, N, SO> B;
        StaticMatrix<Real, M, N, SO> X;

        randomize(A);
        randomize(B);

        for (auto _ : state)
        {
            trsm<UPLO, UNIT>(A, B, X);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(X);
        }

        setCounters(state.counters, complexity(trsmTag, UNIT, M, N));
        state.counters["m"] = M;
        state.counters["n"] = N;
    }


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_TRSM)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_trsm_left_dense_static, double, UpLo::Upper, true, n, n, columnMajor);
#include BOOST_PP_LOCAL_ITERATE()
}
