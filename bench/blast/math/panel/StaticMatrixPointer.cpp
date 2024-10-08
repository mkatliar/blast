// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>

#include <bench/Benchmark.hpp>


namespace blast :: benchmark
{
    template <typename Real, AlignmentFlag AF>
    static void BM_static_panel_matrix_pointer(State& state)
    {
        size_t constexpr M = 8;
        size_t constexpr N = 4;

        StaticPanelMatrix<Real, M, N, columnMajor> A;
        auto pA = ptr<AF>(A, 0, 0);

        for (auto _ : state)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    DoNotOptimize(pA(i, j));
        }
    }


    BENCHMARK_TEMPLATE(BM_static_panel_matrix_pointer, double, aligned);
    BENCHMARK_TEMPLATE(BM_static_panel_matrix_pointer, double, unaligned);
    BENCHMARK_TEMPLATE(BM_static_panel_matrix_pointer, float, aligned);
    BENCHMARK_TEMPLATE(BM_static_panel_matrix_pointer, float, unaligned);
}
