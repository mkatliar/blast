// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/Matrix.hpp>
#include <blast/math/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <blast/math/algorithm/Randomize.hpp>

#include <blast/blaze/Math.hpp>


namespace blast :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO, UpLo UPLO>
    static void BM_RegisterMatrix_trmmLeft(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t constexpr K = 100;

        blaze::StaticMatrix<T, Kernel::rows(), Kernel::rows(), SO> A;
        blaze::StaticMatrix<T, Kernel::rows(), Kernel::columns(), columnMajor> B;

        randomize(A);
        randomize(B);

        Kernel ker;

        for (auto _ : state)
        {
            ker.trmm(T(1.), ptr(A), UPLO, false, ptr(B));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(M * N * (M + 1), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, bool SO, UpLo UPLO>
    static void BM_RegisterMatrix_trmmRight(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t constexpr K = 100;

        blaze::StaticMatrix<T, Kernel::columns(), Kernel::columns(), SO> A;
        blaze::StaticMatrix<T, Kernel::rows(), Kernel::columns(), columnMajor> B;

        randomize(A);
        randomize(B);

        Kernel ker;

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
