// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_ger_nt(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using ET = ElementType_t<Kernel>;
        size_t constexpr K = 100;

        DynamicPanelMatrix<ET, columnMajor> a(Kernel::rows(), K);
        DynamicPanelMatrix<ET, columnMajor> b(Kernel::columns(), K);
        DynamicPanelMatrix<ET, columnMajor> c(Kernel::rows(), Kernel::columns());

        randomize(a);
        randomize(b);
        randomize(c);

        Kernel ker;
        ker.load(ptr(c));

        for (auto _ : state)
        {
            for (size_t i = 0; i < K; ++i)
                ker.ger(ET(0.1), column(ptr<aligned>(a, 0, i)), row(ptr<aligned>(b, 0, i).trans()));

            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 4, 8, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 12, 4, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 24, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 5, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 6, columnMajor);
}
