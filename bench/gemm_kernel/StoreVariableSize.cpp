#include <smoke/StaticMatrix.hpp>
#include <smoke/GemmKernel_double_1_1_4.hpp>
#include <smoke/GemmKernel_double_2_1_4.hpp>
#include <smoke/GemmKernel_double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>

#include <functional> 


namespace smoke :: benchmark
{
    template <typename Kernel>
    static void BM_GemmKernel_storeVariableSize(State& state)
    {
        using Traits = GemmKernelTraits<Kernel>;
        size_t constexpr M = Traits::rows;
        size_t constexpr N = Traits::columns;
        size_t const m = state.range(0);
        size_t const n = state.range(1);

        StaticMatrix<double, M, N, Traits::blockSize, Traits::alignment> c, d;
        randomize(c);

        Kernel ker(c.block(0, 0), c.spacing());

        for (auto _ : state)
        {
            ker.store(d.block(0, 0), d.spacing(), m, n);
            DoNotOptimize(d);   
        }

        state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
    }


    static void args(internal::Benchmark * b, size_t M, size_t N) 
    {
        for (int i = 1; i <= M; ++i)
            for (int j = 1; j <= N; ++j)
                b->Args({i, j});
    }


    static void args_4_4(internal::Benchmark * b) 
    {
        args(b, 4, 4);
    }


    static void args_8_4(internal::Benchmark * b) 
    {
        args(b, 8, 4);
    }


    static void args_12_4(internal::Benchmark * b) 
    {
        args(b, 12, 4);
    }


    using std::placeholders::_1;

    BENCHMARK_TEMPLATE(BM_GemmKernel_storeVariableSize, GemmKernel<double, 1, 1, 4, false, true>)->Apply(args_4_4);
    BENCHMARK_TEMPLATE(BM_GemmKernel_storeVariableSize, GemmKernel<double, 2, 1, 4, false, true>)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_GemmKernel_storeVariableSize, GemmKernel<double, 3, 1, 4, false, true>)->Apply(args_12_4);
}