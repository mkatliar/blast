// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// @brief Eigen GEMM benchmarks.
///

// Allow Eigen static matrices of any size to be created on stack.
#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <Eigen/Dense>

#include <test/Randomize.hpp>

#include <bench/Gemm.hpp>


namespace blast :: benchmark
{
    using namespace ::benchmark;
    using blast::testing::randomize;


    template <typename Real, size_t M>
    static void BM_gemm_static(::benchmark::State& state)
    {
        size_t constexpr N = M;
        size_t constexpr K = M;

        Eigen::Matrix<Real, K, M, Eigen::ColMajor> A;
        A.setRandom();

        Eigen::Matrix<Real, K, N, Eigen::ColMajor> B;
        B.setRandom();

        Eigen::Matrix<Real, M, N, Eigen::ColMajor> C, D;
        C.setRandom();

        Real alpha, beta;
        randomize(alpha);
        randomize(beta);

        for (auto _ : state)
        {
            D = alpha * A.transpose() * B + beta * C;
            ::benchmark::DoNotOptimize(A);
            ::benchmark::DoNotOptimize(B);
            ::benchmark::DoNotOptimize(C);
            ::benchmark::DoNotOptimize(D);
        }

        setCounters(state.counters, complexityGemm(M, N, K));
        state.counters["m"] = M;
        state.counters["n"] = N;
        state.counters["k"] = K;
    }


    template <typename Real>
    static void BM_gemm_dynamic(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A(m, m);
        A.setRandom();

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> B(m, m);
        B.setRandom();

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> C(m, m), D(m, m);
        C.setRandom();

        Real alpha, beta;
        randomize(alpha);
        randomize(beta);

        for (auto _ : state)
        {
            D = alpha * A.transpose() * B + beta * C;
            ::benchmark::DoNotOptimize(A);
            ::benchmark::DoNotOptimize(B);
            ::benchmark::DoNotOptimize(C);
            ::benchmark::DoNotOptimize(D);
        }

        setCounters(state.counters, complexityGemm(m, m, m));
        state.counters["m"] = m;
    }


    BENCHMARK_TEMPLATE(BM_gemm_dynamic, double)->DenseRange(1, BENCHMARK_MAX_GEMM);


#define BOOST_PP_LOCAL_LIMITS (1, BENCHMARK_MAX_GEMM)
#define BOOST_PP_LOCAL_MACRO(N) \
    BENCHMARK_TEMPLATE(BM_gemm_static, double, N); \
    BENCHMARK_TEMPLATE(BM_gemm_static, float, N);
#include BOOST_PP_LOCAL_ITERATE()
}
