#include <smoke/StaticPanelMatrix.hpp>
#include <smoke/gemm/GemmKernel_double_1_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_2_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    template <typename Kernel>
    static void BM_GemmKernel_store(State& state)
    {
        using Traits = GemmKernelTraits<Kernel>;
        size_t constexpr M = Traits::rows;
        size_t constexpr N = Traits::columns;

        StaticPanelMatrix<double, M, N, Traits::blockSize, rowMajor, Traits::alignment> c, d;
        randomize(c);

        Kernel ker(c.block(0, 0), c.spacing());

        for (auto _ : state)
        {
            ker.store(d.block(0, 0), d.spacing());
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(M * N, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_GemmKernel_store, GemmKernel<double, 1, 1, 4, false, true>);
    BENCHMARK_TEMPLATE(BM_GemmKernel_store, GemmKernel<double, 2, 1, 4, false, true>);
    BENCHMARK_TEMPLATE(BM_GemmKernel_store, GemmKernel<double, 3, 1, 4, false, true>);
}