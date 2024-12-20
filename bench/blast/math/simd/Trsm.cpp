// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Trsm.hpp>

#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_trsm(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;

        StaticPanelMatrix<double, M, N, columnMajor> L;
        randomize(L);

        StaticPanelMatrix<double, M, N, columnMajor> A;
        randomize(A);

        Kernel ker;
        ker.load(ptr(A));

        for (auto _ : state)
        {
            ker.trsm(Side::Left, UpLo::Upper, ptr(L).trans());
            DoNotOptimize(ker);
        }

        setCounters(state.counters, complexity(trsmTag, false, M, N));
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 8, 8, columnMajor);
}
