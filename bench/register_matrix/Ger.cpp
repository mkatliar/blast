#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t P>
    static void BM_RegisterMatrix_ger_nt(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, P>;
        using Traits = RegisterMatrixTraits<Kernel>;
        using ET = ElementType_t<Kernel>;
        size_t constexpr K = 100;
        
        DynamicPanelMatrix<ET> a(Traits::rows, K);
        DynamicPanelMatrix<ET> b(Traits::columns, K);
        DynamicPanelMatrix<ET> c(Traits::rows, Traits::columns);

        randomize(a);
        randomize(b);
        randomize(c);

        Kernel ker;
        load(ker, c.ptr(0, 0), c.spacing());

        for (auto _ : state)
        {
            for (size_t i = 0; i < K; ++i)
                ger<a.storageOrder, !decltype(b)::storageOrder>(ker, ET(0.1), ptr(a, 0, i), spacing(a), ptr(b, 0, i), spacing(b));

            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(2 * M * N * K, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 4, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 4, 8, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 8, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, double, 12, 4, 4);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 8, 4, 8);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 4, 8);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 24, 4, 8);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 5, 8);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_ger_nt, float, 16, 6, 8);
}