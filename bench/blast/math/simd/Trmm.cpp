// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/Matrix.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/dense/StaticMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, StorageOrder SO, UpLo UPLO>
    static void BM_RegisterMatrix_trmmLeft(State& state)
    {
        size_t constexpr K = 100;

        StaticMatrix<T, M, M, SO> A;
        StaticMatrix<T, M, N, columnMajor> B;

        randomize(A);
        randomize(B);

        RegisterMatrix<T, M, N, SO> ker;

        for (auto _ : state)
        {
            ker.trmm(T(1.), ptr(A), UPLO, false, ptr(B));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(M * N * (M + 1), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, StorageOrder SO, UpLo UPLO>
    static void BM_RegisterMatrix_trmmRight(State& state)
    {
        size_t constexpr K = 100;

        StaticMatrix<T, N, N, SO> A;
        StaticMatrix<T, M, N, columnMajor> B;

        randomize(A);
        randomize(B);

        RegisterMatrix<T, M, N, SO> ker;

        for (auto _ : state)
        {
            ker.trmm(T(1.), ptr(A), ptr(B), UPLO, false);
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(N * M * (N + 1), Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, double, 4, 4, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, double, 4, 8, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, double, 8, 4, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, double, 12, 4, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, float, 8, 4, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, float, 16, 4, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, float, 24, 4, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, float, 16, 5, columnMajor, UpLo::Upper);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeft, float, 16, 6, columnMajor, UpLo::Upper);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, double, 4, 4, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, double, 4, 8, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, double, 8, 4, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, double, 12, 4, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, float, 8, 4, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, float, 16, 4, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, float, 24, 4, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, float, 16, 5, columnMajor, UpLo::Lower);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRight, float, 16, 6, columnMajor, UpLo::Lower);
}
