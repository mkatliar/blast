// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Syrk.hpp>

#include <blaze/Math.h>


#define BENCHMARK_STATIC(M, N) \
    BENCHMARK_TEMPLATE(BM_syrk_static, double, M, N, columnMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_static, double, M, N, rowMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_declsym_static, double, M, N, columnMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_declsym_static, double, M, N, rowMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_symmetric_declsym_static, double, M, N, columnMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_symmetric_declsym_static, double, M, N, rowMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_loop_static, double, M, N, columnMajor); \
    BENCHMARK_TEMPLATE(BM_syrk_loop_static, double, M, N, rowMajor);


namespace blast :: benchmark
{
    using namespace blaze;


    template <typename Real, bool SO>
    static void BM_syrk_dynamic(State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        DynamicMatrix<Real, SO> A(M, N);
        DynamicMatrix<Real, SO> B(N, N);

        randomize(A);

        for (auto _ : state)
            DoNotOptimize(B = trans(A) * A);
    }


    template <typename Real, bool SO>
    static void BM_syrk_symmetric_dynamic(State& state)
    {
        size_t const M = state.range(0);
        size_t const N = state.range(1);
        DynamicMatrix<Real, SO> A(M, N);
        SymmetricMatrix<DynamicMatrix<Real, SO>> B(N);

        randomize(A);

        for (auto _ : state)
            DoNotOptimize(B = declsym(trans(A) * A));
    }


    template <typename Real, size_t M, size_t N, bool SO>
    static void BM_syrk_static(State& state)
    {
        StaticMatrix<Real, M, N, SO> A;
        StaticMatrix<Real, N, N, SO> B;

        randomize(A);

        for (auto _ : state)
            DoNotOptimize(B = trans(A) * A);
    }


    template <typename Real, size_t M, size_t N, bool SO>
    static void BM_syrk_declsym_static(State& state)
    {
        StaticMatrix<Real, M, N, SO> A;
        StaticMatrix<Real, N, N, SO> B;

        randomize(A);

        for (auto _ : state)
            DoNotOptimize(B = declsym(trans(A) * A));
    }


    template <typename Real, size_t M, size_t N, bool SO>
    static void BM_syrk_symmetric_declsym_static(State& state)
    {
        StaticMatrix<Real, M, N, SO> A;
        SymmetricMatrix<StaticMatrix<Real, N, N, SO>> B;

        randomize(A);

        for (auto _ : state)
            DoNotOptimize(B = declsym(trans(A) * A));
    }


    template <typename Real, size_t M, size_t N, bool SO1, bool SO2>
    void syrkStaticLoop(StaticMatrix<Real, M, N, SO1> const& A, StaticMatrix<Real, N, N, SO2>& B)
    {
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t i = j; i < N; ++i)
                B(i, j) = dot(column(A, i), column(A, j));
        }
    }


    template <typename Real, size_t M, size_t N, bool SO>
    static void BM_syrk_loop_static(State& state)
    {
        StaticMatrix<Real, M, N, SO> A;
        StaticMatrix<Real, N, N, SO> B;

        randomize(A);

        for (auto _ : state)
        {
            syrkStaticLoop(A, B);
            DoNotOptimize(A);
            DoNotOptimize(B);
        }
    }


    BENCHMARK_TEMPLATE(BM_syrk_dynamic, double, columnMajor)->Apply(syrkBenchArguments);
    BENCHMARK_TEMPLATE(BM_syrk_symmetric_dynamic, double, columnMajor)->Apply(syrkBenchArguments);

    // BENCHMARK_STATIC(4, 5);
    // BENCHMARK_STATIC(20, 40);
    // BENCHMARK_STATIC(30, 35);
#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_SYRK)
#define BOOST_PP_LOCAL_MACRO(N) \
    BENCHMARK_STATIC(N, N);
#include BOOST_PP_LOCAL_ITERATE()
}
