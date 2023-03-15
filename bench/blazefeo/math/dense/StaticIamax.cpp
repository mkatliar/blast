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

#include <blazefeo/math/dense/Iamax.hpp>

#include <blaze/math/DynamicVector.h>

#include <bench/Iamax.hpp>
#include <bench/Complexity.hpp>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t N>
    static void BM_iamax_static(State& state)
    {
        StaticVector<Real, N> x;
        randomize(x);

        size_t idx;

        for (auto _ : state)
        {
            x[0] = 0;
            idx = iamax(x);
            DoNotOptimize(idx);
        }

        setCounters(state.counters, complexityIamax(N));
    }


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_IAMAX)
#define BOOST_PP_LOCAL_MACRO(n) \
    BENCHMARK_TEMPLATE(BM_iamax_static, double, n); \
    BENCHMARK_TEMPLATE(BM_iamax_static, float, n);
#include BOOST_PP_LOCAL_ITERATE()
}
