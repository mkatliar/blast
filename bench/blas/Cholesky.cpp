// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.



#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_Cholesky_Blaze_potrf(::benchmark::State& state)
    {
        size_t const N = state.range(0);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(N, N);
        makePositiveDefinite(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(N, N);
        for (auto _ : state)
        {
            B = A;
            potrf(B, 'L');
        }
    }


    template <typename Real>
    static void BM_Cholesky_Blaze_llh_Dynamic(::benchmark::State& state)
    {
        size_t const N = state.range(0);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(N, N);
        makePositiveDefinite(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(N, N);
        for (auto _ : state)
            blaze::llh(A, B);
    }


    template <typename Real>
    static void BM_Cholesky_Blaze_llh_SymmetricDynamic(::benchmark::State& state)
    {
        size_t const N = state.range(0);
        blaze::SymmetricMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> A(N);
        makePositiveDefinite(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(N, N);
        for (auto _ : state)
            blaze::llh(A, B);
    }


    template <typename Real>
    static void BM_Cholesky_Blaze_llh_SymmetricToLowerDynamic(::benchmark::State& state)
    {
        size_t const N = state.range(0);
        blaze::SymmetricMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> A(N);
        makePositiveDefinite(A);

        blaze::LowerMatrix<blaze::DynamicMatrix<Real, blaze::columnMajor>> B(N);
        for (auto _ : state)
            blaze::llh(A, B);
    }


    template <typename Real>
    static void BM_Cholesky_Blaze_llh_DeclsymToDynamic(::benchmark::State& state)
    {
        size_t const N = state.range(0);
        blaze::DynamicMatrix<Real, blaze::columnMajor> A(N, N);
        makePositiveDefinite(A);

        blaze::DynamicMatrix<Real, blaze::columnMajor> B(N, N);
        for (auto _ : state)
            blaze::llh(declsym(A), B);
    }


    static void choleskyBenchArguments(::benchmark::internal::Benchmark* b)
    {
        b->Arg(1)->Arg(2)->Arg(5)->Arg(10)->Arg(35);
    }


    BENCHMARK_TEMPLATE(BM_Cholesky_Blaze_llh_Dynamic, double)->Apply(choleskyBenchArguments);
    BENCHMARK_TEMPLATE(BM_Cholesky_Blaze_llh_SymmetricDynamic, double)->Apply(choleskyBenchArguments);
    BENCHMARK_TEMPLATE(BM_Cholesky_Blaze_llh_SymmetricToLowerDynamic, double)->Apply(choleskyBenchArguments);
    BENCHMARK_TEMPLATE(BM_Cholesky_Blaze_llh_DeclsymToDynamic, double)->Apply(choleskyBenchArguments);
    BENCHMARK_TEMPLATE(BM_Cholesky_Blaze_potrf, double)->Apply(choleskyBenchArguments);
}
