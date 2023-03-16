// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.



#include <benchmark/benchmark.h>

#include <blaze/Math.h>


namespace tmpc :: benchmark
{
    template <typename Real, size_t M, size_t N>
    static void BM_syrkPotrf_static(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> A;
        blaze::SymmetricMatrix<blaze::StaticMatrix<Real, N, N, blaze::columnMajor>> C;
        blaze::StaticMatrix<Real, N, N, blaze::columnMajor> D;

        randomize(A);
        makePositiveDefinite(C);

        for (auto _ : state)
        {
            D = C;
            cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                N, M, 1., data(A), spacing(A), 1., data(D), spacing(D));
            potrf(D, 'L');
        }
    }


    template <typename Real, size_t M, size_t N>
    static void BM_syrkPotrf_blaze_static(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> A;
        blaze::SymmetricMatrix<blaze::StaticMatrix<Real, N, N, blaze::columnMajor>> C;
        blaze::SymmetricMatrix<blaze::StaticMatrix<Real, N, N, blaze::columnMajor>> D1;
        blaze::LowerMatrix<blaze::StaticMatrix<Real, N, N, blaze::columnMajor>> D;

        randomize(A);
        makePositiveDefinite(C);

        for (auto _ : state)
        {
            D1 = declsym(C) + declsym(trans(A) * A);
            blaze::llh(D1, D);
        }
    }


    BENCHMARK_TEMPLATE(BM_syrkPotrf_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrkPotrf_static, double, 30, 60);

    BENCHMARK_TEMPLATE(BM_syrkPotrf_blaze_static, double, 4, 5);
    BENCHMARK_TEMPLATE(BM_syrkPotrf_blaze_static, double, 30, 60);
}
