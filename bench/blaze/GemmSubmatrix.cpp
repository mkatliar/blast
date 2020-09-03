// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/Blaze.hpp>

#include <bench/Benchmark.hpp>

#include <vector>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t M>
    static void BM_gemm_submatrix_static(State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;
        
        blaze::StaticMatrix<Real, K, M, blaze::columnMajor> A;
        randomize(A);

        blaze::StaticMatrix<Real, K, N, blaze::columnMajor> B;
        randomize(B);

        blaze::StaticMatrix<Real, M, N, blaze::columnMajor> C;
        randomize(C);
        
        for (auto _ : state)
        {
            submatrix(C, 0, 0, M, N) += trans(submatrix(A, 0, 0, K, M)) * submatrix(B, 0, 0, K, N);
            // blaze::submatrix<0, 0, M, N>(C) += trans(blaze::submatrix<0, 0, K, M>(A)) * blaze::submatrix<0, 0, K, N>(B);
            DoNotOptimize(A);
            DoNotOptimize(B);
            DoNotOptimize(C);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
        state.counters["m"] = M;
        state.counters["n"] = N;
        state.counters["k"] = K;
    }


#define BOOST_PP_LOCAL_LIMITS (1, 10)
#define BOOST_PP_LOCAL_MACRO(N) \
    BENCHMARK_TEMPLATE(BM_gemm_submatrix_static, double, N);
#include BOOST_PP_LOCAL_ITERATE()
}
