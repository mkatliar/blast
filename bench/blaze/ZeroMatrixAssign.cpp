// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.



#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    template <typename Real, size_t M, size_t N>
    static void BM_zeroMatrixAssign_dynamic(::benchmark::State& state)
    {
        blaze::DynamicMatrix<Real> A(M, N);

        for (auto _ : state)
            ::benchmark::DoNotOptimize(A = blaze::ZeroMatrix<Real>(M, N));
    }


    template <typename Real, size_t M, size_t N>
    static void BM_DynamicMatrixZeroAssign(::benchmark::State& state)
    {
        blaze::DynamicMatrix<Real> A(M, N);

        for (auto _ : state)
            ::benchmark::DoNotOptimize(A = Real {0});
    }


    BENCHMARK_TEMPLATE(BM_zeroMatrixAssign_dynamic, double, 4, 1);
    BENCHMARK_TEMPLATE(BM_DynamicMatrixZeroAssign, double, 4, 1);
}
