// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_trmmLeftUpper(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t constexpr K = 100;

        StaticMatrix<T, Kernel::rows(), Kernel::rows(), SO> A;
        StaticMatrix<T, Kernel::rows(), Kernel::columns(), columnMajor> B;

        randomize(A);
        randomize(B);

        Kernel ker;

        for (auto _ : state)
        {
            ker.trmmLeftUpper(T(1.), ptr<aligned>(A, 0, 0), ptr<aligned>(B, 0, 0));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(M * N * (M + 1), Counter::kIsIterationInvariantRate);
    }


    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_trmmRightLower(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t constexpr K = 100;

        StaticMatrix<T, Kernel::columns(), Kernel::columns(), SO> A;
        StaticMatrix<T, Kernel::rows(), Kernel::columns(), columnMajor> B;

        randomize(A);
        randomize(B);

        Kernel ker;

        for (auto _ : state)
        {
            ker.trmmRightLower(T(1.), ptr<aligned>(A, 0, 0), ptr<aligned>(B, 0, 0));
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(N * M * (N + 1), Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, double, 4, 8, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, float, 24, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, float, 16, 5, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmLeftUpper, float, 16, 6, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, double, 4, 8, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, double, 12, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, float, 24, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, float, 16, 5, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmmRightLower, float, 16, 6, columnMajor);
}