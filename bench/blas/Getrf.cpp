// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Complexity.hpp>
#include <bench/Benchmark.hpp>

#include <blazefeo/Blaze.hpp>

#include <vector>


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;


    template <typename Real>
    static void BM_getrf(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        std::vector<int> ipiv(m);

        // Using the identity matrix as an input to getrf().
        // Since it does not change after the getrf(), we don't need to fill the matrix with proper data before each call,
        // which removes the performance overhead.
        // Also it provides the best case for the getrf() implementation, because no row permutations are needed.
        // This gives us a stable performance of the benchmarked function.
        blaze::DynamicMatrix<Real, blaze::columnMajor> A = blaze::IdentityMatrix<double>(m);

        for (auto _ : state)
            getrf(A, ipiv.data());

        setCounters(state.counters, complexityGetrf(rows(A), columns(A)));
        state.counters["m"] = m;
    }

    BENCHMARK_TEMPLATE(BM_getrf, double)->DenseRange(1, 50);
    BENCHMARK_TEMPLATE(BM_getrf, float)->DenseRange(1, 50);
}
