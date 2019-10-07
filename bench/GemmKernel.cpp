#include <smoke/GemmKernel.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    template <bool TA, bool TB>
    static void BM_GemmKernel(State& state)
    {
        size_t constexpr N = 4;

        alignas(GemmKernel<double, 1, 1, N>::alignment) std::array<double, N * N> a, b, c;
        randomize(a);
        randomize(b);
        randomize(c);

        GemmKernel<double, 1, 1, N> kc;
        kc.load(c.data(), c.size());

        for (auto _ : state)
        {
            kc(a.data(), TA, b.data(), TB);
            DoNotOptimize(kc);
        }

        state.counters["flops"] = Counter(N * N * N, Counter::kIsIterationInvariantRate);
        state.counters["m"] = N;
    }


    BENCHMARK_TEMPLATE(BM_GemmKernel, true, false);
    BENCHMARK_TEMPLATE(BM_GemmKernel, false, false);
    BENCHMARK_TEMPLATE(BM_GemmKernel, false, true);
}