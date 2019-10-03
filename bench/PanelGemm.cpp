#include <smoke/Panel.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    static void BM_PanelGemm(State& state)
    {
        size_t constexpr N = 4;

        alignas(Panel<double, N>::alignment) std::array<double, N * N> a, b, c;
        randomize(a);
        randomize(b);
        randomize(c);

        Panel<double, N> pa, pb, pc;
        pa.load(a.data());
        pb.load(b.data());
        pc.load(c.data());

        for (auto _ : state)
        {
            gemm(pa, true, pb, false, pc);
            DoNotOptimize(pc);
        }

        state.counters["flops"] = Counter(N * N * N, Counter::kIsIterationInvariantRate);
        state.counters["m"] = N;
    }


    BENCHMARK(BM_PanelGemm);
}