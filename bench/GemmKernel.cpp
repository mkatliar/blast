#include <smoke/StaticMatrix.hpp>
#include <smoke/GemmKernel_double_1_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    template <typename Kernel, bool TA, bool TB>
    static void BM_GemmKernel(State& state)
    {
        using Traits = GemmKernelTraits<Kernel>;

        StaticMatrix<double, Traits::rows, Traits::blockSize, Traits::blockSize, Traits::alignment> a;
        StaticMatrix<double, Traits::columns, Traits::blockSize, Traits::blockSize, Traits::alignment> b;
        StaticMatrix<double, Traits::rows, Traits::columns, Traits::blockSize, Traits::alignment> c, d;

        randomize(a);
        randomize(b);
        randomize(c);

        Kernel kc;
        kc.load(c.block(0, 0), c.spacing());

        for (auto _ : state)
        {
            kc(a.block(0, 0), a.spacing(), TA, b.block(0, 0), b.spacing(), TB);
            DoNotOptimize(kc);
        }

        state.counters["flops"] = Counter(Traits::rows * Traits::blockSize * Traits::columns, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_GemmKernel, GemmKernel<double, 1, 1, 4>, true, false);
    BENCHMARK_TEMPLATE(BM_GemmKernel, GemmKernel<double, 1, 1, 4>, false, false);
    BENCHMARK_TEMPLATE(BM_GemmKernel, GemmKernel<double, 1, 1, 4>, false, true);
}