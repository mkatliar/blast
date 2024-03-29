// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <bench/Benchmark.hpp>
#include <test/Randomize.hpp>

#include <libxsmm.h>

#include <vector>


namespace blast :: benchmark
{
    template <typename Real>
    static void BM_gemm_nn(::benchmark::State& state)
    {
        using value_type = double;

        size_t const m = state.range(0);
        size_t const n = m;
        size_t const k = m;
        std::vector<value_type> a(m * k), b(k * n), c(m * n, 0);

        /* C/C++ and Fortran interfaces are available */
        using kernel_type = libxsmm_mmfunction<value_type>;

        /* generates and dispatches a matrix multiplication kernel (C++ functor) */
        kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m, n, k, 1.0/*alpha*/, 1.0/*beta*/);
        assert(kernel);

        for (auto& aa : a)
            blaze::randomize(aa);

        for (auto& bb : b)
            blaze::randomize(bb);

        /* kernel multiplies and accumulates matrix products: C += Ai * Bi */
        for (auto _ : state)
            kernel(a.data(), b.data(), c.data());

        state.counters["flops"] = Counter(m * n * k, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    template <typename Real>
    static void BM_gemm_nt(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        size_t const n = m;
        size_t const k = m;
        std::vector<Real> a(m * k), b(n * k), c(m * n, 0);

        /* C/C++ and Fortran interfaces are available */
        using kernel_type = libxsmm_mmfunction<Real>;

        /* generates and dispatches a matrix multiplication kernel (C++ functor) */
        kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, m, n, k, 1.0/*alpha*/, 1.0/*beta*/);
        assert(kernel);

        for (auto& aa : a)
            blaze::randomize(aa);

        for (auto& bb : b)
            blaze::randomize(bb);

        /* kernel multiplies and accumulates matrix products: C += Ai * Bi */
        for (auto _ : state)
            kernel(a.data(), b.data(), c.data());

        state.counters["flops"] = Counter(2 * m * n * k, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }


    BENCHMARK_TEMPLATE(BM_gemm_nn, double)->DenseRange(1, 50);
    BENCHMARK_TEMPLATE(BM_gemm_nt, double)->DenseRange(1, 50);

    BENCHMARK_TEMPLATE(BM_gemm_nn, float)->DenseRange(1, 50);
    BENCHMARK_TEMPLATE(BM_gemm_nt, float)->DenseRange(1, 50);
}
