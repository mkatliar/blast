#include <smoke/StaticPanelMatrix.hpp>
#include <smoke/gemm/GemmKernel_double_1_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_2_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    template <typename Kernel>
    static void BM_GemmKernel_gemm(State& state)
    {
        using Traits = GemmKernelTraits<Kernel>;
        size_t constexpr M = Traits::rows;
        size_t constexpr N = Traits::columns;
        size_t constexpr K = 1;
        bool constexpr TA = Traits::tA;
        bool constexpr TB = Traits::tB;

        StaticPanelMatrix<double, !TA ? M : K, !TA ? K : M, Traits::blockSize, Traits::alignment> a;
        StaticPanelMatrix<double, !TB ? K : N, !TB ? N : K, Traits::blockSize, Traits::alignment> b;
        StaticPanelMatrix<double, M, N, Traits::blockSize, Traits::alignment> c, d;

        randomize(a);
        randomize(b);
        randomize(c);

        Kernel ker(c.block(0, 0), c.spacing());

        for (auto _ : state)
        {
            ker(a.block(0, 0), a.spacing(), b.block(0, 0), b.spacing());
            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(M * N * K, Counter::kIsIterationInvariantRate);
    }


    // BENCHMARK_TEMPLATE(BM_GemmKernel_gemm, GemmKernel<double, 1, 1, 4, true, false>);
    // BENCHMARK_TEMPLATE(BM_GemmKernel_gemm, GemmKernel<double, 1, 1, 4, false, false>);
    BENCHMARK_TEMPLATE(BM_GemmKernel_gemm, GemmKernel<double, 1, 1, 4, false, true>);
    
    // BENCHMARK_TEMPLATE(BM_GemmKernel_gemm, GemmKernel<double, 2, 1, 4, true, false>);
    BENCHMARK_TEMPLATE(BM_GemmKernel_gemm, GemmKernel<double, 2, 1, 4, false, true>);

    BENCHMARK_TEMPLATE(BM_GemmKernel_gemm, GemmKernel<double, 3, 1, 4, false, true>);
}