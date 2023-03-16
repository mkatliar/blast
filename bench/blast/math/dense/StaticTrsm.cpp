// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/UpLo.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/math/dense/Trsm.hpp>

#include <blaze/math/StaticMatrix.h>

#include <bench/Trsm.hpp>


namespace blast :: benchmark
{
    template <typename Real, UpLo UPLO, bool UNIT, size_t M, size_t N, StorageOrder SO>
    static void BM_trsm_left_dense_static(State& state)
    {
        blaze::StaticMatrix<Real, M, M, SO> A;
        blaze::StaticMatrix<Real, M, N, SO> B;
        blaze::StaticMatrix<Real, M, N, SO> X;

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
    BENCHMARK_TEMPLATE(BM_trsm_left_dense_static, double, UpLo::Upper, false, n, n, columnMajor);
#include BOOST_PP_LOCAL_ITERATE()
}
