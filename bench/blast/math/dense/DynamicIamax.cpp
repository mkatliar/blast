// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <blast/math/dense/Iamax.hpp>

#include <blaze/math/DynamicVector.h>

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
