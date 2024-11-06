// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_potrf(State& state)
    {
        DynamicPanelMatrix<T> a(M, N);
        randomize(a);

        RegisterMatrix<T, M, N, SO> ker;
        ker.load(ptr(a));

        for (auto _ : state)
        {
            ker.potrf();
            DoNotOptimize(ker);
        }

        if (M >= N)
            setCounters(state.counters, complexityPotrf(M, N));
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 12, 4, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 24, 4, columnMajor);
}
