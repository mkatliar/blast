// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/dense/Getrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>

#include <vector>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_getrf_static_plain(State& state)
    {
        StaticMatrix<Real, M, M, columnMajor> A = IdentityMatrix<Real>(M);
        std::vector<size_t> ipiv(M);

        for (auto _ : state)
        {
            getrf(A, ipiv.data());
            DoNotOptimize(A);
        }

        setCounters(state.counters, complexityGetrf(M, M));
        state.counters["m"] = M;
    }


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GETRF)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_getrf_static_plain, double, n);\
    // BENCHMARK_TEMPLATE(BM_getrf_static_plain, float, n);
#include BOOST_PP_LOCAL_ITERATE()
}
