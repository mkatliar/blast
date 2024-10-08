// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/panel/Potrf.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <blast/math/algorithm/Randomize.hpp>

#include <random>
#include <memory>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_potrf_dynamic_panel(State& state)
    {
        size_t const M = state.range(0);

        DynamicPanelMatrix<Real, columnMajor> A(M, M), L(M, M);
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


    BENCHMARK_TEMPLATE(BM_potrf_dynamic_panel, double)->DenseRange(1, BENCHMARK_MAX_POTRF);
    BENCHMARK_TEMPLATE(BM_potrf_dynamic_panel, float)->DenseRange(1, BENCHMARK_MAX_POTRF);
}
