// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <Eigen/Dense>

#include <benchmark/benchmark.h>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_potrf_dynamic(::benchmark::State& state)
    {
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
        size_t const N = state.range(0);

        // Make a positive definite matrix A
        Matrix A = Matrix::Random(N, N);
        A = A.transpose() * A;

        Eigen::LLT<Matrix> llt(N);
        for (auto _ : state)
            ::benchmark::DoNotOptimize(llt.compute(A));
    }


    template <typename Real>
    static void BM_potrf_selfadjoint_dynamic(::benchmark::State& state)
    {
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
        size_t const N = state.range(0);

        // Make a positive definite matrix A
        Matrix A = Matrix::Random(N, N);
        A = A.transpose() * A;

        Eigen::LLT<Matrix> llt(N);
        for (auto _ : state)
            ::benchmark::DoNotOptimize(llt.compute(A.template selfadjointView<Eigen::Lower>()));
    }


    template <typename Real, int N>
    static void BM_potrf_static(::benchmark::State& state)
    {
        using Matrix = Eigen::Matrix<Real, N, N>;

        // Make a positive definite matrix A
        Matrix A = Matrix::Random();
        A = A.transpose() * A;

        Eigen::LLT<Matrix> llt(N);
        for (auto _ : state)
            ::benchmark::DoNotOptimize(llt.compute(A));
    }


    static void choleskyBenchArguments(::benchmark::internal::Benchmark* b)
    {
        b->Arg(1)->Arg(2)->Arg(5)->Arg(10)->Arg(35);
    }


    BENCHMARK_TEMPLATE(BM_potrf_dynamic, double)->Apply(choleskyBenchArguments);
    BENCHMARK_TEMPLATE(BM_potrf_selfadjoint_dynamic, double)->Apply(choleskyBenchArguments);
    BENCHMARK_TEMPLATE(BM_potrf_static, double, 1);
    BENCHMARK_TEMPLATE(BM_potrf_static, double, 2);
    BENCHMARK_TEMPLATE(BM_potrf_static, double, 5);
    BENCHMARK_TEMPLATE(BM_potrf_static, double, 10);
    BENCHMARK_TEMPLATE(BM_potrf_static, double, 35);
}
