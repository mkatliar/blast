// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/dense/StaticMatrixPointer.hpp>
#include <blast/math/dense/StaticMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <blast/math/algorithm/Randomize.hpp>

#include <functional>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_partialStore_panel(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t const m = state.range(0);
        size_t const n = state.range(1);

        Kernel ker;

        DynamicPanelMatrix<T> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
        randomize(c);

        ker.load(ptr(c));
        for (auto _ : state)
        {
            ker.store(ptr(d), m, n);
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
        state.counters["store_m"] = m;
        state.counters["store_n"] = n;
        state.counters["size_m"] = ker.rows();
        state.counters["size_n"] = ker.columns();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_partialStore(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t const m = state.range(0);
        size_t const n = state.range(1);

        Kernel ker;

        blaze::StaticMatrix<T, M, N, SO> C, D;
        randomize(C);

        ker.load(1., ptr<aligned>(C, 0, 0));
        for (auto _ : state)
        {
            ker.store(ptr<aligned>(D, 0, 0), m, n);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
        state.counters["store_m"] = m;
        state.counters["store_n"] = n;
        state.counters["size_m"] = ker.rows();
        state.counters["size_n"] = ker.columns();
    }


    template <typename T, size_t M, size_t N, StorageOrder SO, size_t MM, size_t NN>
    static void BM_RegisterMatrix_partialStore_static(State& state)
    {
        RegisterMatrix<T, M, N, SO> ker;

        blaze::StaticMatrix<T, M, N, SO> C, D;
        randomize(C);

        ker.load(1., ptr<aligned>(C, 0, 0));
        for (auto _ : state)
        {
            ker.store(ptr<aligned>(D, 0, 0), MM, NN);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(MM * NN, Counter::kIsIterationInvariantRate);
        state.counters["store_m"] = MM;
        state.counters["store_n"] = NN;
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


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_panel, double, 4, 4, columnMajor)->Apply(args_4_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_panel, double, 8, 4, columnMajor)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_panel, double, 12, 4, columnMajor)->Apply(args_12_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_panel, float, 8, 4, columnMajor)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_panel, float, 16, 4, columnMajor)->Apply(args_16_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_panel, float, 24, 4, columnMajor)->Apply(args_24_4);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 4, 4, columnMajor)->Apply(args_4_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 8, 4, columnMajor)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 12, 4, columnMajor)->Apply(args_12_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, float, 8, 4, columnMajor)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, float, 16, 4, columnMajor)->Apply(args_16_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, float, 24, 4, columnMajor)->Apply(args_24_4);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_static, double, 4, 4, columnMajor, 1, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore_static, double, 4, 4, columnMajor, 2, 4);
}
