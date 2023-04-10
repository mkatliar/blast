// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Trsm.hpp>

#include <test/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_trsm(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;
        size_t constexpr m = Traits::rows;
        size_t constexpr n = Traits::columns;

        StaticPanelMatrix<double, m, n, columnMajor> L;
        randomize(L);

        StaticPanelMatrix<double, m, n, columnMajor> A;
        randomize(A);

        Kernel ker;
        ker.load(ptr(A));

        for (auto _ : state)
        {
            ker.trsm(Side::Left, UpLo::Upper, ptr(L).trans());
            DoNotOptimize(ker);
        }

        setCounters(state.counters, complexity(trsmTag, false, m, n));
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 8, 8, columnMajor);
}