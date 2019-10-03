#include <smoke/Panel.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace smoke :: benchmark
{
    template <bool TA, bool TB, typename T, size_t N>
    void gemm_impl(Panel<T, N> const& a, Panel<T, N> const& b, Panel<T, N>& c);


    template <bool TA, bool TB>
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
            gemm(pa, TA, pb, TB, pc);
            DoNotOptimize(pc);
        }

        state.counters["flops"] = Counter(N * N * N, Counter::kIsIterationInvariantRate);
        state.counters["m"] = N;
    }


    BENCHMARK_TEMPLATE(BM_PanelGemm, true, false);
    BENCHMARK_TEMPLATE(BM_PanelGemm, false, false);
}