// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <Eigen/Dense>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real>
    static void BM_syrk_dynamic(::benchmark::State& state)
    {
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

        size_t const M = state.range(0);
        size_t const N = state.range(1);
        Matrix const A = Matrix::Random(M, N);        
        Matrix B(N, N);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = A.transpose() * A);
    }


    template <typename Real>
    static void BM_syrk_rankUpdate_dynamic(::benchmark::State& state)
    {
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

        size_t const M = state.range(0);
        size_t const N = state.range(1);
        Matrix const A = Matrix::Random(M, N);        
        Matrix B(N, N);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B.template selfadjointView<Eigen::Lower>().rankUpdate(A.transpose()));
    }


    template <typename Real>
    static void BM_syrk_triangular_dynamic(::benchmark::State& state)
    {
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

        size_t const M = state.range(0);
        size_t const N = state.range(1);
        Matrix const A = Matrix::Random(M, N);        
        Matrix B(N, N);
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B.template triangularView<Eigen::Lower>() = A.transpose() * A);
    }


    template <typename Real, int M, int N>
    static void BM_syrk_static(::benchmark::State& state)
    {
        using MatrixA = Eigen::Matrix<Real, M, N>;
        using MatrixB = Eigen::Matrix<Real, N, N>;

        MatrixA const A = MatrixA::Random();
        MatrixB B;
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B = A.transpose() * A);
    }


    template <typename Real, int M, int N>
    static void BM_syrk_rankUpdate_static(::benchmark::State& state)
    {
        using MatrixA = Eigen::Matrix<Real, M, N>;
        using MatrixB = Eigen::Matrix<Real, N, N>;

        MatrixA const A = MatrixA::Random();
        MatrixB B;
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B.template selfadjointView<Eigen::Lower>().rankUpdate(A.transpose()));
    }


    template <typename Real, int M, int N>
    static void BM_syrk_triangular_static(::benchmark::State& state)
    {
        using MatrixA = Eigen::Matrix<Real, M, N>;
        using MatrixB = Eigen::Matrix<Real, N, N>;

        MatrixA const A = MatrixA::Random();
        MatrixB B;
        
        for (auto _ : state)
            ::benchmark::DoNotOptimize(B.template triangularView<Eigen::Lower>() = A.transpose() * A);
    }


    static void syrkBenchArguments(::benchmark::internal::Benchmark* b) 
    {
        b->Args({4, 5})->Args({30, 35});
    }


    BENCHMARK_TEMPLATE(BM_syrk_dynamic, double)->Apply(syrkBenchArguments);
    BENCHMARK_TEMPLATE(BM_syrk_rankUpdate_dynamic, double)->Apply(syrkBenchArguments);
    BENCHMARK_TEMPLATE(BM_syrk_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_static, double, 30, 35);
    BENCHMARK_TEMPLATE(BM_syrk_rankUpdate_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_rankUpdate_static, double, 30, 35);
    BENCHMARK_TEMPLATE(BM_syrk_triangular_dynamic, double)->Apply(syrkBenchArguments);
    BENCHMARK_TEMPLATE(BM_syrk_triangular_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrk_triangular_static, double, 30, 35);
}
