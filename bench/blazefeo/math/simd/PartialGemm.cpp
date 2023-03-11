// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/dense/StaticMatrixPointer.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO, size_t MM, size_t NN>
    static void BM_RegisterMatrix_partialGemm_static(State& state)
    {
        size_t constexpr K = 5;

        StaticMatrix<T, M, K, SO> A;
        StaticMatrix<T, K, N, SO> B;
        StaticMatrix<T, M, N, SO> C, D;

        randomize(A);
        randomize(B);
        randomize(C);

        RegisterMatrix<T, M, N, SO> ker;
        for (auto _ : state)
        {
            gemm(ker, K, 1., ptr<aligned>(A, 0, 0), ptr<aligned>(B, 0, 0), 1., ptr<aligned>(C, 0, 0), ptr<aligned>(D, 0, 0), MM, NN);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
            DoNotOptimize(D);
        }

        state.counters["flops"] = Counter(2 * MM * NN * K, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialGemm_static, double, 4, 4, columnMajor, 1, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialGemm_static, double, 4, 4, columnMajor, 2, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialGemm_static, double, 4, 4, columnMajor, 3, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialGemm_static, double, 4, 4, columnMajor, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialGemm_static, double, 4, 4, columnMajor, 1, 1);
}