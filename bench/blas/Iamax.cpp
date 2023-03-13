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

#include <cblas.h>

#include <blaze/math/DynamicVector.h>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    namespace
    {
        inline size_t iamax(size_t n, double const * x, size_t incx)
        {
            return cblas_idamax(n, x, 1);
        }


        inline size_t iamax(size_t n, float const * x, size_t incx)
        {
            return cblas_isamax(n, x, 1);
        }
    };


    template <typename Real>
    static void BM_iamax(State& state)
    {
        size_t const N = 100;
        DynamicVector<Real> x(N);
        randomize(x);

        for (auto _ : state)
        {
            size_t idx = iamax(size(x), data(x), 1);
            DoNotOptimize(idx);
        }

        setCounters(state.counters, complexityIamax(N));
    }


    BENCHMARK_TEMPLATE(BM_iamax, double);
    BENCHMARK_TEMPLATE(BM_iamax, float);
}
