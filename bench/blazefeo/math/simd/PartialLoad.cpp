// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>

#include <functional>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_partialLoad_static_dense(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t const m = state.range(0);
        size_t const n = state.range(1);

        Kernel ker;

        StaticMatrix<T, M, N, SO> C;
        randomize(C);

        ker.load(1., ptr<aligned>(C, 0, 0));
        for (auto _ : state)
        {
            ker.load(1., ptr<aligned>(C, 0, 0), m, n);
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
        state.counters["store_m"] = m;
        state.counters["store_n"] = n;
        state.counters["size_m"] = ker.rows();
        state.counters["size_n"] = ker.columns();
    }


    // template <typename T, size_t M, size_t N, size_t SS>
    // static void BM_RegisterMatrix_partialStore_panelAverage(State& state)
    // {
    //     using Kernel = RegisterMatrix<T, M, N, SS>;
    //     Kernel ker;

    //     DynamicPanelMatrix<double, rowMajor> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
    //     randomize(c);

    //     load(ker, c.ptr<aligned>(0, 0), c.spacing());

    //     for (int m = 1; m <= ker.rows(); ++m)
    //         for (int n = 1; n <= ker.columns(); ++n)
    //             for (auto _ : state)
    //             {
    //                 store(ker, d.ptr<aligned>(0, 0), d.spacing(), m, n);
    //                 DoNotOptimize(d);
    //             }

    //     state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
    // }


    static void args(internal::Benchmark * b, size_t M, size_t N)
    {
        for (int i = 1; i <= M; ++i)
            for (int j = 1; j <= N; ++j)
                b->Args({i, j});
    }


    static void args_4_4(internal::Benchmark * b)
    {
        args(b, 4, 4);
    }


    static void args_8_4(internal::Benchmark * b)
    {
        args(b, 8, 4);
    }


    static void args_12_4(internal::Benchmark * b)
    {
        args(b, 12, 4);
    }


    static void args_16_4(internal::Benchmark * b)
    {
        args(b, 16, 4);
    }


    static void args_24_4(internal::Benchmark * b)
    {
        args(b, 24, 4);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialLoad_static_dense, double, 4, 4, columnMajor)->Apply(args_4_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialLoad_static_dense, double, 8, 4, columnMajor)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialLoad_static_dense, double, 12, 4, columnMajor)->Apply(args_12_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialLoad_static_dense, float, 8, 4, columnMajor)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialLoad_static_dense, float, 16, 4, columnMajor)->Apply(args_16_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialLoad_static_dense, float, 24, 4, columnMajor)->Apply(args_24_4);

    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 4, 4, columnMajor)->Apply(args_4_4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 8, 4, columnMajor)->Apply(args_8_4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 12, 4, columnMajor)->Apply(args_12_4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, float, 8, 4, columnMajor)->Apply(args_8_4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, float, 16, 4, columnMajor)->Apply(args_16_4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, float, 24, 4, columnMajor)->Apply(args_24_4);

    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_static, double, 4, 4, columnMajor, 1, 4);
    // BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_static, double, 4, 4, columnMajor, 2, 4);
}