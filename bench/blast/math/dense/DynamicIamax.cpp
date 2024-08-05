// Copyright (c) 2023-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/dense/Iamax.hpp>
#include <blast/blaze/Math.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Iamax.hpp>
#include <bench/Complexity.hpp>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_iamax_dynamic(State& state)
    {
        size_t const N = state.range(0);
        DynamicVector<Real> x(N);
        randomize(x);

        size_t idx;

        for (auto _ : state)
        {
            x[0] = 0.;
            idx = iamax(x);
            DoNotOptimize(idx);
        }

        setCounters(state.counters, complexity(iamaxTag, N));
    }


    BENCHMARK_TEMPLATE(BM_iamax_dynamic, double)->DenseRange(1, BENCHMARK_MAX_IAMAX_DYNAMIC);
    BENCHMARK_TEMPLATE(BM_iamax_dynamic, float)->DenseRange(1, BENCHMARK_MAX_IAMAX_DYNAMIC);
}
