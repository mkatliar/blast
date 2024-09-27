// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#include <blast/math/dense/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <blast/math/algorithm/Randomize.hpp>

#include <random>
#include <memory>


namespace blast :: benchmark
{
    template <typename Real, size_t M>
    static void BM_potrf_static_plain(State& state)
    {
        StaticMatrix<Real, M, M, columnMajor> A, L;
        makePositiveDefinite(A);

        for (auto _ : state)
        {
            potrf(A, L);
            DoNotOptimize(A);
            DoNotOptimize(L);
        }

        setCounters(state.counters, complexityPotrf(M, M));
        state.counters["m"] = M;
    }


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_POTRF)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_potrf_static_plain, double, n);\
    BENCHMARK_TEMPLATE(BM_potrf_static_plain, float, n);
#include BOOST_PP_LOCAL_ITERATE()
}
