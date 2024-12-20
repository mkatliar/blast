// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/dense/DynamicMatrixPointer.hpp>
#include <blast/math/dense/StaticMatrixPointer.hpp>

#include <bench/Benchmark.hpp>

#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_load_dynamic_panel(State& state)
    {
        RegisterMatrix<T, M, N, SO> ker;

        DynamicPanelMatrix<double> c(ker.rows(), ker.columns());
        randomize(c);

        for (auto _ : state)
        {
            ker.load(ptr(c));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_load_dynamic_dense(State& state)
    {
        RegisterMatrix<T, M, N, SO> ker;

        DynamicMatrix<double, SO> c(ker.rows(), ker.columns());
        randomize(c);

        for (auto _ : state)
        {
            ker.load(ptr<aligned>(c, 0, 0));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, StorageOrder SO>
    static void BM_RegisterMatrix_load_static_dense(State& state)
    {
        RegisterMatrix<T, M, N, SO> ker;

        StaticMatrix<double, M, N, SO> c;
        randomize(c);

        for (auto _ : state)
        {
            ker.load(ptr<aligned>(c, 0, 0));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_panel, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_panel, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_panel, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_panel, double, 8, 8, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_dense, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_dense, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_dense, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_dynamic_dense, double, 8, 8, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_static_dense, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_static_dense, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_static_dense, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_load_static_dense, double, 8, 8, columnMajor);
}
