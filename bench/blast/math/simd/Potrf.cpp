// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_potrf(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;
        size_t constexpr m = Traits::rows;
        size_t constexpr n = Traits::columns;

        DynamicPanelMatrix<T> a(m, n);
        randomize(a);

        Kernel ker;
        ker.load(ptr(a));

        for (auto _ : state)
        {
            ker.potrf();
            DoNotOptimize(ker);
        }

        if (m >= n)
            setCounters(state.counters, complexityPotrf(m, n));
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 12, 4, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 24, 4, columnMajor);
}