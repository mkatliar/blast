// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_store_dynamic_panel(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;

        Kernel ker;

        DynamicPanelMatrix<double> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
        randomize(c);

        load(ker, c.ptr(0, 0), c.spacing());

        for (auto _ : state)
        {
            store(ker, d.ptr(0, 0), d.spacing());
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_store_dynamic_dense(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;

        Kernel ker;

        DynamicMatrix<double, SO> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
        randomize(c);

        ker.load(ptr<aligned>(c, 0, 0));

        for (auto _ : state)
        {
            ker.store(ptr<aligned>(d, 0, 0));
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_store_static_dense(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;

        Kernel ker;

        StaticMatrix<double, M, N, SO> c, d;
        randomize(c);

        ker.load(ptr<aligned>(c, 0, 0));

        for (auto _ : state)
        {
            ker.store(ptr<aligned>(d, 0, 0));
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_panel, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_panel, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_panel, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_panel, double, 8, 8, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_dense, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_dense, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_dense, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_dynamic_dense, double, 8, 8, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_static_dense, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_static_dense, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_static_dense, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store_static_dense, double, 8, 8, columnMajor);
}