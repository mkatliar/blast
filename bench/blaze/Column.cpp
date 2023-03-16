// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.



#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace blast :: benchmark
{
    template <typename Real, size_t N, bool CL, bool CR>
    static void BM_column(::benchmark::State& state)
    {
        blaze::StaticMatrix<double, N, N, blaze::columnMajor> A;
        randomize(A);

        for (auto _ : state)
        {
            size_t const k = N / 2;
            size_t const rs = N - k;

            auto D21 = submatrix(A, k, k, rs, 1, blaze::checked);
            auto const D20 = submatrix(A, k, 0, rs, k, blaze::checked);

            for (size_t j = 0; j < k; ++j)
                column(D21, 0, blaze::Check<CL> {}) -= (*A)(k, j) * column(D20, j, blaze::Check<CR> {});

            ::benchmark::DoNotOptimize(A(N - 1, N - 1));
        }
    }


    BENCHMARK_TEMPLATE(BM_column, double, 60, false, false);
    BENCHMARK_TEMPLATE(BM_column, double, 60, false, true);
    BENCHMARK_TEMPLATE(BM_column, double, 60, true, false);
    BENCHMARK_TEMPLATE(BM_column, double, 60, true, true);
}
