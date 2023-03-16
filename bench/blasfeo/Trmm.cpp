// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blasfeo/Blasfeo.hpp>

#include <bench/Benchmark.hpp>

#include <random>
#include <memory>


#define ADD_BM_TRMM(m, n, p) BENCHMARK_CAPTURE(BM_trmm, m##x##n##x##p##_blasfeo, m, n, p)


namespace blast :: benchmark
{
    template <typename MT>
    static void randomize(blasfeo::Matrix<MT>& A)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (size_t i = 0; i < rows(~A); ++i)
            for (size_t j = 0; j < columns(~A); ++j)
                (~A)(i, j) = dis(gen);
    }


    static void BM_trmm(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        size_t const n = state.range(1);

        blasfeo::DynamicMatrix<double> A(m, m), B(m, n), C(m, n);

        randomize(A);
        randomize(B);

        for (auto _ : state)
            trmm_rutn(m, n, 1., A, 0, 0, B, 0, 0, C, 0, 0);
    }


    BENCHMARK(BM_trmm)->Apply(trmmBenchArguments);
}
