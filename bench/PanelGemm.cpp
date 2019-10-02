#include <smoke/Panel.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    static void BM_PanelGemm(State& state)
    {
        size_t constexpr N = 4;

        std::array<double, N * N> a, b, c;
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
    }


    BENCHMARK(BM_PanelGemm);
}